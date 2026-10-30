import React, { useState, useRef, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";
import * as pdfjsLib from 'pdfjs-dist';

// Initialize PDF.js worker
// Using esm.sh for the worker ensures version compatibility with the library imported from esm.sh
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://esm.sh/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

// --- Stage 1: Scene-change detection settings ---
const SAMPLE_INTERVAL_SEC = 1; // Sample roughly every second
const MAX_SAMPLES = 7200; // Long videos sample more sparsely to keep this many samples at most
const SIGNATURE_WIDTH = 64; // Small grayscale signature used for frame differencing
const SIGNATURE_HEIGHT = 36;
const DEFAULT_DIFF_THRESHOLD = 5; // Mean absolute pixel difference (% of 255) that counts as a slide change
const MIN_DIFF_THRESHOLD = 1;
const MAX_DIFF_THRESHOLD = 30;
const STABLE_THRESHOLD = 1.5; // Consecutive frames differing less than this (%) are considered settled
const MAX_SETTLE_SAMPLES = 5; // Give up waiting for the picture to settle after this many samples
const MAX_CANDIDATES = 600; // Refuse to send more candidates than this (raise the threshold instead)
const SEEK_TIMEOUT_MS = 5000;

// --- Stage 2: Gemini matching settings ---
const CANDIDATE_WIDTH = 768; // Fits in a single Gemini image tile while keeping slide text readable
const BATCH_SIZE = 30; // Candidate frames per Gemini request

interface PdfImage {
  data: string; // Base64
  page: number;
}

interface VideoSignatures {
  file: File;
  duration: number;
  interval: number;
  times: number[]; // Sample times in seconds
  sigs: Uint8Array[]; // Grayscale SIGNATURE_WIDTH x SIGNATURE_HEIGHT pixels per sample
}

interface Candidate {
  id: string;
  seconds: number; // Detected change time (midpoint between the previous and changed sample)
  frameSeconds: number; // Time of the settled representative frame
}

interface CandidateFrame extends Candidate {
  data: string; // Base64 JPEG of the representative frame
}

interface ModelMatch {
  candidateId: string;
  isSlide: boolean;
  pdfId: number;
  pageNumber: number;
  slideTitle: string;
  reasoning: string;
  confidence: Confidence;
}

type Confidence = "High" | "Medium" | "Low";

interface SlideMatch {
  timestamp: string;
  seconds: number; // Program-detected change time, used for jumping in the video player
  frameTimestamp: string;
  frameSeconds: number;
  thumbnail: string; // Data URL of the candidate frame
  pdfThumbnail: string | null; // Data URL of the matched PDF page
  pdfId: number;
  pageNumber: number;
  slideTitle: string;
  reasoning: string;
  confidence: Confidence;
  mergedCount: number; // Number of candidate frames merged into this event
  flagged: boolean; // Failed a consistency check, needs manual review
}

// Errors with a message that is already user-facing (Traditional Chinese) and should be shown as-is
class UserFacingError extends Error {}

const CONFIDENCE_LABELS: Record<string, string> = { High: '高', Medium: '中', Low: '低' };
const CONFIDENCE_RANK: Record<string, number> = { Low: 0, Medium: 1, High: 2 };

// --- Helper: Time formatting (MM:SS, or HH:MM:SS for videos of 1 hour or longer) ---
const formatTime = (seconds: number, useHours: boolean): string => {
  const total = Math.max(0, Math.floor(seconds));
  const hh = Math.floor(total / 3600).toString().padStart(2, '0');
  const mm = Math.floor((useHours ? total % 3600 : total) / 60).toString().padStart(2, '0');
  const ss = (total % 60).toString().padStart(2, '0');
  return useHours ? `${hh}:${mm}:${ss}` : `${mm}:${ss}`;
};

// --- Helper: Parse model JSON, repairing truncated output by dropping the last incomplete item ---
const parseJsonArrayField = (text: string, field: string): { items: any[]; truncated: boolean } => {
  let cleanText = text.replace(/```json/g, '').replace(/```/g, '').trim();
  const jsonStart = cleanText.indexOf('{');
  if (jsonStart > 0) cleanText = cleanText.substring(jsonStart);

  try {
    const json = JSON.parse(cleanText);
    return { items: Array.isArray(json[field]) ? json[field] : [], truncated: false };
  } catch {
    // Fall through to truncation repair
  }

  // Expected structure: { "<field>": [ {...}, {...}, ... ]
  // Scan while tracking strings and nesting depth to find the end of the last complete item
  // (an item's closing brace brings depth back to 2: root object -> array).
  let depth = 0;
  let inString = false;
  let escaped = false;
  let lastCompleteItemEnd = -1;
  for (let i = 0; i < cleanText.length; i++) {
    const ch = cleanText[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{' || ch === '[') depth++;
    else if (ch === '}' || ch === ']') {
      depth--;
      if (ch === '}' && depth === 2) lastCompleteItemEnd = i;
    }
  }

  if (lastCompleteItemEnd === -1) {
    if (cleanText.includes(`"${field}"`)) {
      console.warn("Response truncated before any complete item. Returning empty list.");
      return { items: [], truncated: true };
    }
    throw new UserFacingError("無法解析分析結果：模型回應不是有效的 JSON 格式。");
  }

  const repaired = cleanText.substring(0, lastCompleteItemEnd + 1) + ']}';
  try {
    const json = JSON.parse(repaired);
    console.warn("Response appeared truncated. Discarded the last incomplete item.");
    return { items: Array.isArray(json[field]) ? json[field] : [], truncated: true };
  } catch {
    throw new UserFacingError("無法解析分析結果：模型回應被截斷且無法修復。");
  }
};

// --- Helper: Map errors to Traditional Chinese user-facing messages ---
const toErrorMessage = (error: unknown): string => {
  if (error instanceof UserFacingError) return error.message;

  const msg = error instanceof Error ? error.message : String(error);
  const status = typeof (error as any)?.status === 'number' ? (error as any).status : undefined;

  if (status === 413 || /\b413\b|too large|payload size|exceeds the limit/i.test(msg)) {
    return "檔案過大：送出的影格與投影片總量超過 API 請求大小上限。請減少 PDF 頁數或壓縮檔案後再試。";
  }
  if (status === 400 || /\b400\b|INVALID_ARGUMENT/.test(msg)) {
    return `請求失敗 (400)：輸入內容可能超出模型 token 上限。請減少 PDF 頁數後再試。\n原始訊息：${msg}`;
  }
  if (status === 401 || status === 403 || /API key|PERMISSION_DENIED/i.test(msg)) {
    return `API 金鑰無效或沒有權限，請檢查 AI Studio 的 API Key 設定。\n原始訊息：${msg}`;
  }
  if (status === 429 || /\b429\b|RESOURCE_EXHAUSTED|quota/i.test(msg)) {
    return "已達 API 使用量上限 (429)，請稍後再試。";
  }
  if ((status !== undefined && status >= 500) || /\b50[0-9]\b|UNAVAILABLE|overloaded/i.test(msg)) {
    return `Gemini 服務暫時無法使用，請稍後再試。\n原始訊息：${msg}`;
  }
  return `分析過程中發生未預期的錯誤：${msg}`;
};

// --- Helper: Video loading and seeking ---
const loadVideo = (file: File): Promise<HTMLVideoElement> => {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;
    video.src = URL.createObjectURL(file);

    video.onloadeddata = () => {
      if (!Number.isFinite(video.duration) || video.duration <= 0 || !video.videoWidth) {
        URL.revokeObjectURL(video.src);
        reject(new UserFacingError("影格抽取失敗：無法讀取影片長度或畫面尺寸，請確認影片檔案完整。"));
        return;
      }
      resolve(video);
    };
    video.onerror = () => {
      URL.revokeObjectURL(video.src);
      reject(new UserFacingError("影格抽取失敗：瀏覽器無法載入這支影片。請確認格式為瀏覽器可播放的 MP4/WebM，且檔案未損壞。"));
    };
  });
};

const releaseVideo = (video: HTMLVideoElement) => {
  URL.revokeObjectURL(video.src);
  video.removeAttribute('src');
  video.load();
};

// Resolves true once the video has seeked, false on timeout
const seekTo = (video: HTMLVideoElement, time: number): Promise<boolean> => {
  return new Promise((resolve) => {
    const cleanup = () => {
      window.clearTimeout(timer);
      video.removeEventListener('seeked', onSeeked);
    };
    const onSeeked = () => {
      cleanup();
      resolve(true);
    };
    const timer = window.setTimeout(() => {
      cleanup();
      resolve(false);
    }, SEEK_TIMEOUT_MS);
    video.addEventListener('seeked', onSeeked);
    video.currentTime = time;
  });
};

// --- Stage 1a: Sample the video into small grayscale signatures ---
const sampleVideoSignatures = async (
  file: File,
  onProgress: (done: number, total: number) => void
): Promise<VideoSignatures> => {
  const video = await loadVideo(file);
  try {
    const duration = video.duration;
    const interval = Math.max(SAMPLE_INTERVAL_SEC, duration / MAX_SAMPLES);
    const total = Math.ceil(duration / interval);

    const canvas = document.createElement('canvas');
    canvas.width = SIGNATURE_WIDTH;
    canvas.height = SIGNATURE_HEIGHT;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new UserFacingError("影格抽取失敗：瀏覽器不支援 Canvas。");

    const times: number[] = [];
    const sigs: Uint8Array[] = [];
    for (let n = 0; n < total; n++) {
      const time = n * interval;
      if (!(await seekTo(video, time))) {
        console.warn(`Seek to ${time}s timed out, skipping sample`);
        continue;
      }
      ctx.drawImage(video, 0, 0, SIGNATURE_WIDTH, SIGNATURE_HEIGHT);
      const rgba = ctx.getImageData(0, 0, SIGNATURE_WIDTH, SIGNATURE_HEIGHT).data;
      const gray = new Uint8Array(SIGNATURE_WIDTH * SIGNATURE_HEIGHT);
      for (let p = 0; p < gray.length; p++) {
        gray[p] = 0.299 * rgba[p * 4] + 0.587 * rgba[p * 4 + 1] + 0.114 * rgba[p * 4 + 2];
      }
      times.push(time);
      sigs.push(gray);
      if (n % 10 === 0) onProgress(n + 1, total);
    }

    if (sigs.length === 0) {
      throw new UserFacingError("影格抽取失敗：無法從影片取得任何畫面。請確認影片格式為瀏覽器可播放的 MP4/WebM，且檔案未損壞。");
    }
    return { file, duration, interval, times, sigs };
  } finally {
    releaseVideo(video);
  }
};

// Mean absolute pixel difference as a percentage of the full 0-255 range
const frameDiff = (a: Uint8Array, b: Uint8Array): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return (sum / a.length / 255) * 100;
};

// --- Stage 1b: Find candidate slide changes ---
// A change is detected when a sample differs from the last representative frame by more than the
// threshold (comparing against the representative rather than the previous sample also catches slow
// fades). The representative is the first sample after the change whose next sample is nearly identical.
const detectCandidates = (signatures: VideoSignatures, threshold: number): Candidate[] => {
  const { sigs, times, interval } = signatures;
  if (sigs.length === 0) return [];
  const stable = Math.min(STABLE_THRESHOLD, threshold / 2);
  const last = sigs.length - 1;

  const settle = (start: number): number => {
    const limit = Math.min(start + MAX_SETTLE_SAMPLES, last);
    for (let j = start; j < limit; j++) {
      if (frameDiff(sigs[j], sigs[j + 1]) <= stable) return j;
    }
    return limit;
  };

  const candidates: Candidate[] = [];
  const addCandidate = (changeIndex: number, frameIndex: number) => {
    // The change may have been triggered by motion before the new picture arrived (e.g. a moving
    // speaker shot followed by a slide), so the onset is the earliest sample already matching the
    // representative frame.
    let onset = frameIndex;
    while (onset > changeIndex && frameDiff(sigs[onset - 1], sigs[frameIndex]) <= threshold) onset--;
    candidates.push({
      id: `c${candidates.length + 1}`,
      seconds: onset === 0 ? 0 : Math.max(0, times[onset] - interval / 2),
      frameSeconds: times[frameIndex],
    });
  };

  let rep = settle(0);
  addCandidate(0, rep);
  let reference = sigs[rep];

  let i = rep + 1;
  while (i <= last) {
    if (frameDiff(sigs[i], reference) > threshold) {
      rep = settle(i);
      // Skip brief flashes that return to the previous picture
      if (frameDiff(sigs[rep], reference) > threshold) {
        addCandidate(i, rep);
        reference = sigs[rep];
      }
      i = rep + 1;
    } else {
      i++;
    }
  }
  return candidates;
};

// --- Stage 1c: Capture representative frames for the candidates ---
const captureCandidateFrames = async (
  file: File,
  candidates: Candidate[],
  onProgress: (done: number, total: number) => void
): Promise<CandidateFrame[]> => {
  const video = await loadVideo(file);
  try {
    const canvas = document.createElement('canvas');
    canvas.width = CANDIDATE_WIDTH;
    canvas.height = Math.round(CANDIDATE_WIDTH * video.videoHeight / video.videoWidth);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new UserFacingError("影格抽取失敗：瀏覽器不支援 Canvas。");

    const frames: CandidateFrame[] = [];
    for (let n = 0; n < candidates.length; n++) {
      const candidate = candidates[n];
      if (!(await seekTo(video, candidate.frameSeconds))) {
        console.warn(`Seek to ${candidate.frameSeconds}s timed out, dropping candidate ${candidate.id}`);
        continue;
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      frames.push({ ...candidate, data: canvas.toDataURL('image/jpeg', 0.7).split(',')[1] });
      onProgress(n + 1, candidates.length);
    }
    return frames;
  } finally {
    releaseVideo(video);
  }
};

// --- Stage 3: Post-processing ---
const buildResults = (
  frames: CandidateFrame[],
  matches: Map<string, ModelMatch>,
  pdf1Images: PdfImage[],
  pdf2Images: PdfImage[],
  useHours: boolean
): { results: SlideMatch[]; nonSlideCount: number } => {
  const isDual = pdf2Images.length > 0;
  const results: SlideMatch[] = [];
  let nonSlideCount = 0;

  for (const frame of frames) {
    const match = matches.get(frame.id);
    if (!match) continue;
    if (!match.isSlide) {
      nonSlideCount++;
      continue;
    }

    const pages = match.pdfId === 1 ? pdf1Images : match.pdfId === 2 && isDual ? pdf2Images : null;
    const page = pages?.find(p => p.page === match.pageNumber);

    // Merge consecutive events on the same page (non-slide frames in between do not break the run)
    const previous = results[results.length - 1];
    if (previous && previous.pdfId === match.pdfId && previous.pageNumber === match.pageNumber) {
      previous.mergedCount++;
      if (!previous.flagged && CONFIDENCE_RANK[match.confidence] > CONFIDENCE_RANK[previous.confidence]) {
        previous.confidence = match.confidence;
      }
      continue;
    }

    results.push({
      timestamp: formatTime(frame.seconds, useHours),
      seconds: frame.seconds,
      frameTimestamp: formatTime(frame.frameSeconds, useHours),
      frameSeconds: frame.frameSeconds,
      thumbnail: `data:image/jpeg;base64,${frame.data}`,
      pdfThumbnail: page ? `data:image/jpeg;base64,${page.data}` : null,
      pdfId: match.pdfId,
      pageNumber: match.pageNumber,
      slideTitle: match.slideTitle,
      reasoning: page ? match.reasoning
        : match.pdfId === 2 && !isDual ? `${match.reasoning}（單份簡報模式不應出現 PDF 2，請人工確認）`
        : `${match.reasoning}（模型回傳的頁碼不存在，請人工確認）`,
      confidence: page ? match.confidence : 'Low',
      mergedCount: 1,
      flagged: !page,
    });
  }

  // Dual deck: PDF 1 -> PDF 2 switches exactly once. Pick the split point that violates the fewest
  // events, then mark events on the wrong side of it as low confidence.
  if (isDual && results.length > 0) {
    let cost = results.filter(r => r.pdfId === 1).length; // Split at 0: every PDF 1 event is a violation
    let bestCost = cost;
    let bestSplit = 0;
    for (let k = 1; k <= results.length; k++) {
      cost += results[k - 1].pdfId === 2 ? 1 : -1;
      if (cost < bestCost) {
        bestCost = cost;
        bestSplit = k;
      }
    }
    results.forEach((r, idx) => {
      const expected = idx < bestSplit ? 1 : 2;
      if (r.pdfId !== expected) {
        r.flagged = true;
        r.confidence = 'Low';
        r.reasoning += '（違反「PDF 1 → PDF 2 只切換一次」的順序，請人工確認）';
      }
    });
  }

  return { results, nonSlideCount };
};

const App = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [pdf1File, setPdf1File] = useState<File | null>(null);
  const [pdf2File, setPdf2File] = useState<File | null>(null);
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SlideMatch[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [threshold, setThreshold] = useState(DEFAULT_DIFF_THRESHOLD);
  const [signatures, setSignatures] = useState<VideoSignatures | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Create a preview URL for the selected video and release it when the file changes
  useEffect(() => {
    if (!videoFile) {
      setVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(videoFile);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  // Signatures are only valid for the video they were sampled from
  const currentSignatures = signatures && signatures.file === videoFile ? signatures : null;
  const candidateCount = useMemo(
    () => (currentSignatures ? detectCandidates(currentSignatures, threshold).length : null),
    [currentSignatures, threshold]
  );

  const jumpToTime = (seconds: number) => {
    const video = videoRef.current;
    if (video && Number.isFinite(seconds)) {
      video.currentTime = seconds;
      video.play().catch((e) => console.warn("Video playback was blocked:", e));
      video.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // --- Helper: Extract Images from PDF ---
  const extractPdfImages = async (file: File): Promise<PdfImage[]> => {
    const arrayBuffer = await file.arrayBuffer();
    // Configure CMaps to fix font loading warnings
    // Using unpkg.com because it reliably serves the binary .bcmap files which might be missing or mime-typed incorrectly on other CDNs
    const pdf = await pdfjsLib.getDocument({
      data: arrayBuffer,
      cMapUrl: `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/cmaps/`,
      cMapPacked: true,
    }).promise;
    const images: PdfImage[] = [];

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 1.0 }); // Standard scale for API to read text
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (!context) continue;

      canvas.height = viewport.height;
      canvas.width = viewport.width;

      // Fix: Cast to any to suppress incorrect type definition requiring 'canvas' property
      await page.render({ canvasContext: context, viewport } as any).promise;
      // Use JPEG to save token size
      const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      images.push({ data: base64, page: i });
    }
    return images;
  };

  // --- Stage 2: Match one batch of candidate frames to PDF pages with Gemini ---
  const matchBatchWithGemini = async (
    ai: GoogleGenAI,
    pdf1Images: PdfImage[],
    pdf2Images: PdfImage[],
    batch: CandidateFrame[],
    useHours: boolean,
    previousSlide: string | null
  ): Promise<{ matches: ModelMatch[]; truncated: boolean }> => {
    const isDual = pdf2Images.length > 0;

    // Construct Parts Array based on Requirements
    const parts: any[] = [];

    // 1. PDF 1 Section
    parts.push({ text: "【參考資料 A：第一份簡報 (PDF 1)】\n這是主要演講使用的投影片，依序出現：" });
    pdf1Images.forEach(img => {
      parts.push({
        inlineData: {
          mimeType: "image/jpeg",
          data: img.data
        }
      });
      parts.push({ text: `(PDF1 Page ${img.page})` });
    });

    // 2. PDF 2 Section (Optional)
    // Dynamic Prompt Construction: Only add PDF 2 if images exist
    if (isDual) {
      parts.push({ text: "\n\n【參考資料 B：第二份簡報 (PDF 2)】\n這是演講下半場或補充使用的投影片，若有出現會接續在 PDF 1 之後：" });
      pdf2Images.forEach(img => {
        parts.push({
          inlineData: {
            mimeType: "image/jpeg",
            data: img.data
          }
        });
        parts.push({ text: `(PDF2 Page ${img.page})` });
      });
    }

    // 3. Candidate Frames Section
    parts.push({ text: "\n\n【待比對目標：候選影格】\n以下是程式偵測到畫面切換後擷取的影格，依時間排序，每張附有編號與時間：" });
    batch.forEach(frame => {
      parts.push({ text: `\n[CANDIDATE_ID: ${frame.id}｜時間 ${formatTime(frame.frameSeconds, useHours)}]` });
      parts.push({
        inlineData: {
          mimeType: "image/jpeg",
          data: frame.data
        }
      });
    });

    // 4. System Prompt (Appended at the end)
    const systemPrompt = `
"""
你是一位專業的演講影片分析專家，擅長將「現場演講影片」的畫面與「原始 PDF 投影片」進行視覺比對。

**任務目標：**
針對每一張【候選影格】，判斷它顯示的是哪一份 PDF 的第幾頁，或是「非投影片畫面」。
時間由程式決定，你只需要回答對應關係，不需要判斷時間。

**比對規則 (請嚴格遵守)：**

1.  **非投影片畫面**：
    * 主持人介紹、講者特寫、觀眾、等待畫面、過場動畫、黑畫面等，若畫面中**看不到可辨識的投影片內容**，請將 \`isSlide\` 設為 false。
    * 不要強行匹配：只有在投影片內容與某頁明確相符時才設為 true。

2.  **投影片畫面**：
    * 投影片可能因拍攝角度、講者遮擋、畫中畫縮小或畫質而不完整，請根據文字標題、圖表形狀、圖片排版判斷。
    * 若內容相似的頁面有多頁（例如逐步顯示的條列），請選擇與畫面**完全一致**的那一頁。
    ${isDual ?
      '* 演講順序是先展示 PDF 1，再展示 PDF 2，兩者之間只會切換一次。' :
      '* 只有一份簡報，pdfId 固定為 1。若畫面中的投影片不屬於 PDF 1，請視為非投影片畫面。'}
    ${previousSlide ? `* 前一批候選影格最後對應到的是 ${previousSlide}，可作為順序參考。` : ''}

3.  **輸出格式 (JSON)**：
    請輸出一個 JSON 物件，包含一個 \`matches\` 陣列，**每一張候選影格都要有一筆**，依編號順序排列。
    * \`candidateId\`: 字串，候選影格編號（例如 "c12"）。
    * \`isSlide\`: 布林值，是否為投影片畫面。
    * \`pdfId\`: 整數 (${isDual ? '1 或 2' : '固定為 1'})，非投影片時填 0。
    * \`pageNumber\`: 整數，對應的 PDF 頁碼，非投影片時填 0。
    * \`slideTitle\`: 字串，投影片標題；若無標題請總結畫面核心內容，非投影片時簡述畫面（例如「講者特寫」）。
    * \`reasoning\`: 字串 (正體中文)，簡述判斷理由，請保持簡短。
    * \`confidence\`: 字串 ("High", "Medium", "Low")。
"""
`;
    parts.push({ text: systemPrompt });

    // Call API
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash', // Capable of handling large context and reasoning
      contents: {
        parts: parts
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            matches: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  candidateId: { type: Type.STRING },
                  isSlide: { type: Type.BOOLEAN },
                  pdfId: { type: Type.INTEGER },
                  pageNumber: { type: Type.INTEGER },
                  slideTitle: { type: Type.STRING },
                  reasoning: { type: Type.STRING },
                  confidence: { type: Type.STRING, enum: ["High", "Medium", "Low"] }
                },
                required: ["candidateId", "isSlide", "pdfId", "pageNumber", "slideTitle", "reasoning", "confidence"]
              }
            }
          }
        }
      }
    });

    const responseText = response.text;
    if (!responseText) {
      const blockReason = response.promptFeedback?.blockReason;
      const finishReason = response.candidates?.[0]?.finishReason;
      const detail = blockReason ? `（封鎖原因：${blockReason}）` : finishReason ? `（結束原因：${finishReason}）` : '';
      throw new UserFacingError(`模型回應為空${detail}。可能是內容被安全過濾器攔截，或輸出在產生前就已用盡 token，請稍後再試。`);
    }

    const { items, truncated } = parseJsonArrayField(responseText, 'matches');
    return { matches: items as ModelMatch[], truncated };
  };

  // --- Main Handler ---
  const processFiles = async () => {
    // UI Validation Change: Only require Video and PDF 1
    if (!videoFile || !pdf1File) {
      alert("請至少上傳影片與 PDF 1（主要簡報）。");
      return;
    }

    setIsAnalyzing(true);
    setStatus('正在處理檔案（影片與 PDF）...');
    setError(null);
    setResults([]);

    try {
      // 1. Parallel Processing: PDF rendering and video sampling (reused if the video has not changed)
      const pdf1Promise = extractPdfImages(pdf1File);
      const pdf2Promise = pdf2File ? extractPdfImages(pdf2File) : Promise.resolve([]);
      const signaturesPromise = currentSignatures
        ? Promise.resolve(currentSignatures)
        : sampleVideoSignatures(videoFile, (done, total) => setStatus(`第 1 段：取樣影格 ${done}/${total}...`));

      const [pdf1Images, pdf2Images, videoSignatures] = await Promise.all([
        pdf1Promise,
        pdf2Promise,
        signaturesPromise
      ]);
      setSignatures(videoSignatures);

      if (pdf1Images.length === 0) {
        throw new UserFacingError("PDF 1 轉換失敗：無法取得任何頁面，請確認檔案未損壞。");
      }

      // 2. Scene-change detection
      const useHours = videoSignatures.duration >= 3600;
      const candidates = detectCandidates(videoSignatures, threshold);
      if (candidates.length > MAX_CANDIDATES) {
        throw new UserFacingError(`偵測到 ${candidates.length} 個候選切換點，超過上限 ${MAX_CANDIDATES} 個。請把「切換門檻」調高後再試。`);
      }

      const frames = await captureCandidateFrames(videoFile, candidates, (done, total) =>
        setStatus(`第 1 段：擷取候選影格 ${done}/${total}...`)
      );
      if (frames.length === 0) {
        throw new UserFacingError("影格抽取失敗：無法擷取任何候選影格，請確認影片檔案完整。");
      }

      // 3. Gemini matching in batches; candidates missing from a response are retried once
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const queue: { frames: CandidateFrame[]; isRetry: boolean }[] = [];
      for (let i = 0; i < frames.length; i += BATCH_SIZE) {
        queue.push({ frames: frames.slice(i, i + BATCH_SIZE), isRetry: false });
      }
      const matches = new Map<string, ModelMatch>();
      let previousSlide: string | null = null;

      for (let b = 0; b < queue.length; b++) {
        const batch = queue[b];
        setStatus(`第 2 段：Gemini 比對第 ${b + 1}/${queue.length} 批（${batch.frames.length} 張候選影格）...`);
        const { matches: batchMatches } = await matchBatchWithGemini(
          ai, pdf1Images, pdf2Images, batch.frames, useHours, batch.isRetry ? null : previousSlide
        );

        const batchIds = new Set(batch.frames.map(f => f.id));
        batchMatches.forEach(m => {
          if (m && batchIds.has(m.candidateId)) matches.set(m.candidateId, m);
        });

        const missing = batch.frames.filter(f => !matches.has(f.id));
        if (missing.length > 0 && !batch.isRetry) {
          console.warn(`Batch ${b + 1}: ${missing.length} candidates missing from response, retrying`);
          queue.push({ frames: missing, isRetry: true });
        }
        if (!batch.isRetry) {
          const lastSlide = [...batch.frames].reverse().map(f => matches.get(f.id)).find(m => m?.isSlide);
          if (lastSlide) previousSlide = `PDF ${lastSlide.pdfId} 第 ${lastSlide.pageNumber} 頁`;
        }
      }

      // 4. Post-processing
      const { results: finalResults, nonSlideCount } = buildResults(frames, matches, pdf1Images, pdf2Images, useHours);
      const unmatchedCount = frames.filter(f => !matches.has(f.id)).length;
      const flaggedCount = finalResults.filter(r => r.flagged).length;

      setResults(finalResults);
      const summary = [
        `候選影格 ${frames.length} 張`,
        `非投影片 ${nonSlideCount} 張`,
        flaggedCount > 0 ? `${flaggedCount} 筆標記為待確認` : '',
        unmatchedCount > 0 ? `${unmatchedCount} 張模型未回傳判斷` : '',
      ].filter(Boolean).join('、');
      setStatus(finalResults.length > 0
        ? `分析完成，共 ${finalResults.length} 筆投影片事件（${summary}）。`
        : `未偵測到任何投影片（${summary}）。`);

    } catch (err) {
      console.error(err);
      setStatus('');
      setError(toErrorMessage(err));
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="p-8 max-w-6xl mx-auto font-sans">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">影片投影片分析工具（單份或雙份簡報）</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
          <label className="block text-sm font-semibold text-blue-900 mb-2">1. 影片檔案（MP4/WebM）</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
            className="w-full text-sm text-gray-700"
          />
        </div>

        <div className="p-4 bg-red-50 rounded-lg border border-red-100">
          <label className="block text-sm font-semibold text-red-900 mb-2">PDF 1（主要簡報）</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setPdf1File(e.target.files?.[0] || null)}
            className="w-full text-sm text-gray-700"
          />
        </div>

        <div className="p-4 bg-green-50 rounded-lg border border-green-100">
          <label className="block text-sm font-semibold text-green-900 mb-2">PDF 2（選填）</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setPdf2File(e.target.files?.[0] || null)}
            className="w-full text-sm text-gray-700"
          />
        </div>
      </div>

      <div className="mb-8 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <label htmlFor="threshold" className="flex justify-between text-sm font-semibold text-gray-800 mb-2">
          <span>切換門檻（相鄰畫面平均像素差）</span>
          <span className="font-mono">{threshold.toFixed(1)}%</span>
        </label>
        <input
          id="threshold"
          type="range"
          min={MIN_DIFF_THRESHOLD}
          max={MAX_DIFF_THRESHOLD}
          step={0.5}
          value={threshold}
          disabled={isAnalyzing}
          onChange={(e) => setThreshold(Number(e.target.value))}
          className="w-full"
        />
        <p className="mt-2 text-xs text-gray-600">
          數值越低越敏感：候選影格越多、較不會漏頁，但 Gemini 費用較高；數值越高越可能漏掉內容相近的投影片。預設 {DEFAULT_DIFF_THRESHOLD}%。
          {candidateCount !== null && (
            <span className="ml-1 font-semibold text-gray-800">目前門檻下有 {candidateCount} 個候選切換點。</span>
          )}
        </p>
      </div>

      <div className="mb-8 text-center">
        <button
          onClick={processFiles}
          disabled={isAnalyzing || !videoFile || !pdf1File}
          className={`px-8 py-3 rounded-full text-white font-bold text-lg shadow-lg transition-all ${
            isAnalyzing || !videoFile || !pdf1File
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-xl'
          }`}
        >
          {isAnalyzing ? '分析中...' : '開始分析'}
        </button>
        <p className="mt-4 text-gray-600 font-medium">{status}</p>
        {error && (
          <p className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 font-medium text-left whitespace-pre-line">
            {error}
          </p>
        )}
      </div>

      {videoUrl && (
        <div className="mb-8 bg-black rounded-xl overflow-hidden shadow-md">
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            className="w-full max-h-[500px] object-contain"
          />
        </div>
      )}

      {results.length > 0 && (
        <div className="bg-white rounded-xl shadow-md overflow-x-auto border border-gray-200">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-100 text-gray-700 text-sm uppercase tracking-wider">
                <th className="p-4 border-b">時間</th>
                <th className="p-4 border-b">候選影格</th>
                <th className="p-4 border-b">對應頁面</th>
                <th className="p-4 border-b">來源</th>
                <th className="p-4 border-b">頁碼</th>
                <th className="p-4 border-b">標題／摘要</th>
                <th className="p-4 border-b">信心度</th>
                <th className="p-4 border-b">判斷理由</th>
              </tr>
            </thead>
            <tbody>
              {results.map((match, index) => (
                <tr
                  key={index}
                  className={`border-b last:border-0 transition-colors ${
                    match.flagged ? 'bg-yellow-50 hover:bg-yellow-100' : 'hover:bg-gray-50'
                  }`}
                >
                  <td className="p-4">
                    <button
                      onClick={() => jumpToTime(match.seconds)}
                      title="跳到影片此時間點並播放"
                      className="inline-flex items-center gap-1 px-3 py-1 rounded-lg font-mono font-bold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 transition-colors"
                    >
                      <span aria-hidden="true">▶</span>
                      {match.timestamp}
                    </button>
                  </td>
                  <td className="p-4">
                    <button
                      onClick={() => jumpToTime(match.frameSeconds)}
                      title={`跳到代表影格 ${match.frameTimestamp}`}
                      className="block"
                    >
                      <img src={match.thumbnail} alt={`候選影格 ${match.frameTimestamp}`} className="w-32 rounded border border-gray-300" />
                      <span className="block mt-1 text-xs text-gray-500 font-mono">
                        {match.frameTimestamp}{match.mergedCount > 1 ? `（合併 ${match.mergedCount} 張）` : ''}
                      </span>
                    </button>
                  </td>
                  <td className="p-4">
                    {match.pdfThumbnail ? (
                      <img src={match.pdfThumbnail} alt={`PDF ${match.pdfId} 第 ${match.pageNumber} 頁`} className="w-32 rounded border border-gray-300" />
                    ) : (
                      <span className="text-xs text-gray-400">無預覽</span>
                    )}
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-bold whitespace-nowrap ${
                      match.pdfId === 1 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                    }`}>
                      PDF {match.pdfId}
                    </span>
                  </td>
                  <td className="p-4 font-medium">{match.pageNumber}</td>
                  <td className="p-4 font-semibold text-gray-800">{match.slideTitle}</td>
                  <td className="p-4 whitespace-nowrap">
                    <span className={`inline-block w-3 h-3 rounded-full mr-2 ${
                      match.confidence === 'High' ? 'bg-green-500' :
                      match.confidence === 'Medium' ? 'bg-yellow-500' : 'bg-red-500'
                    }`}></span>
                    {CONFIDENCE_LABELS[match.confidence] ?? match.confidence}
                  </td>
                  <td className="p-4 text-sm text-gray-600 max-w-xs">{match.reasoning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
