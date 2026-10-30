import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";
import * as pdfjsLib from 'pdfjs-dist';

// Initialize PDF.js worker
// Using esm.sh for the worker ensures version compatibility with the library imported from esm.sh
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://esm.sh/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

interface SlideMatch {
  timestamp: string;
  seconds: number; // Corrected time in seconds, used for jumping in the video player
  pdfId: number;
  pageNumber: number;
  slideTitle: string; // New field
  reasoning: string;
  confidence: "High" | "Medium" | "Low";
}

interface PdfImage {
  data: string; // Base64
  page: number;
}

interface VideoFrame {
  data: string; // Base64
  time: number; // Seconds
  timestamp: string;
}

interface VideoFrameResult {
  frames: VideoFrame[];
  interval: number; // Sampling interval in seconds
  useHours: boolean; // Video is 1 hour or longer -> HH:MM:SS
}

// Errors with a message that is already user-facing (Traditional Chinese) and should be shown as-is
class UserFacingError extends Error {}

const CONFIDENCE_LABELS: Record<string, string> = { High: '高', Medium: '中', Low: '低' };

// --- Helper: Time formatting (MM:SS, or HH:MM:SS for videos of 1 hour or longer) ---
const formatTime = (seconds: number, useHours: boolean): string => {
  const total = Math.max(0, Math.floor(seconds));
  const hh = Math.floor(total / 3600).toString().padStart(2, '0');
  const mm = Math.floor((useHours ? total % 3600 : total) / 60).toString().padStart(2, '0');
  const ss = (total % 60).toString().padStart(2, '0');
  return useHours ? `${hh}:${mm}:${ss}` : `${mm}:${ss}`;
};

// --- Helper: Parse MM:SS or HH:MM:SS into seconds (NaN if invalid) ---
const parseTimeToSeconds = (timeStr: string): number => {
  if (!timeStr) return NaN;
  const parts = timeStr.trim().split(':').map(part => Number(part.trim()));
  if (parts.some(p => !Number.isFinite(p))) return NaN;

  if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  } else if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return NaN;
};

// --- Helper: Parse model JSON, repairing truncated output by dropping the last incomplete item ---
const parseTransitionsJson = (text: string): { transitions: any[]; truncated: boolean } => {
  let cleanText = text.replace(/```json/g, '').replace(/```/g, '').trim();
  const jsonStart = cleanText.indexOf('{');
  if (jsonStart > 0) cleanText = cleanText.substring(jsonStart);

  try {
    const json = JSON.parse(cleanText);
    return { transitions: Array.isArray(json.transitions) ? json.transitions : [], truncated: false };
  } catch {
    // Fall through to truncation repair
  }

  // Expected structure: { "transitions": [ {...}, {...}, ... ]
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
    if (cleanText.includes('"transitions"')) {
      console.warn("Response truncated before any complete transition. Returning empty list.");
      return { transitions: [], truncated: true };
    }
    throw new UserFacingError("無法解析分析結果：模型回應不是有效的 JSON 格式。");
  }

  const repaired = cleanText.substring(0, lastCompleteItemEnd + 1) + ']}';
  try {
    const json = JSON.parse(repaired);
    console.warn("Response appeared truncated. Discarded the last incomplete item.");
    return { transitions: Array.isArray(json.transitions) ? json.transitions : [], truncated: true };
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
    return "檔案過大：送出的影格與投影片總量超過 API 請求大小上限。請縮短影片、減少 PDF 頁數或壓縮檔案後再試。";
  }
  if (status === 400 || /\b400\b|INVALID_ARGUMENT/.test(msg)) {
    return `請求失敗 (400)：輸入內容可能超出模型 token 上限。請縮短影片或減少 PDF 頁數後再試。\n原始訊息：${msg}`;
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

const App = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [pdf1File, setPdf1File] = useState<File | null>(null);
  const [pdf2File, setPdf2File] = useState<File | null>(null);
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SlideMatch[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
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

  // --- Helper: Extract Frames from Video ---
  const extractVideoFrames = async (file: File): Promise<VideoFrameResult> => {
    return new Promise((resolve) => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.src = URL.createObjectURL(file);
      video.muted = true;

      const frames: VideoFrame[] = [];
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      video.onloadedmetadata = async () => {
        const duration = video.duration;

        // Validation: Check if duration is valid and finite
        if (!Number.isFinite(duration) || duration <= 0) {
          console.warn("Invalid video duration:", duration);
          URL.revokeObjectURL(video.src);
          resolve({ frames: [], interval: 0, useHours: false });
          return;
        }

        const useHours = duration >= 3600;

        // Sample every 1 second, max 800 frames to fit context
        let interval = 1;
        if (duration / interval > 800) {
            interval = duration / 800;
        }

        canvas.width = 480; // Resize for efficient token usage
        canvas.height = 270;

        for (let time = 0; time < duration; time += interval) {
          await new Promise<void>((r) => {
            // Safety check: Ensure currentTime doesn't exceed duration
            if (time > duration) {
               r();
               return;
            }

            video.currentTime = time;

            video.onseeked = () => {
              if (ctx) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                frames.push({
                  data: canvas.toDataURL('image/jpeg', 0.7).split(',')[1],
                  time,
                  timestamp: formatTime(time, useHours),
                });
              }
              r();
            };

            // Handle seek errors
            video.onerror = () => {
                console.warn(`Error seeking to ${time}s`);
                r();
            };
          });
        }
        URL.revokeObjectURL(video.src);
        resolve({ frames, interval, useHours });
      };

      // Handle general video load errors
      video.onerror = () => {
          console.error("Failed to load video metadata");
          URL.revokeObjectURL(video.src);
          resolve({ frames: [], interval: 0, useHours: false });
      };
    });
  };

  // --- Core: Analyze with Gemini ---
  const analyzeWithGemini = async (
    pdf1Images: PdfImage[],
    pdf2Images: PdfImage[],
    video: VideoFrameResult
  ): Promise<{ transitions: SlideMatch[]; truncated: boolean }> => {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const { frames: videoFrames, interval, useHours } = video;
    const timeFormatLabel = useHours ? 'HH:MM:SS' : 'MM:SS';

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
    if (pdf2Images.length > 0) {
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

    // 3. Video Frames Section
    parts.push({ text: "\n\n【待分析目標：影片影格序列】\n以下是從演講影片中按時間順序取樣的畫面 (帶有時間戳記)：" });
    videoFrames.forEach(frame => {
      parts.push({ text: `\n[VIDEO_TIMESTAMP: ${frame.timestamp}]` });
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
你是一位專業的演講影片分析專家，擅長將「現場演講影片」與「原始 PDF 投影片」進行視覺同步。

**任務目標：**
分析提供的【影片影格序列】，找出每一張投影片（來自 PDF 1${pdf2Images.length > 0 ? ' 或 PDF 2' : ''}）在影片中「首次清晰出現」的時間點。

**核心分析邏輯與規則 (請嚴格遵守)：**

1.  **忽略非投影片畫面 (抗干擾)**：
    * 影片開頭通常包含主持人介紹、講者特寫或等待畫面。請務必等到**投影片內容清晰充滿畫面**，且與 PDF 某頁高度相符時，才標記第一個事件。
    * **切勿強行從 ${formatTime(0, useHours)} 開始**，除非 ${formatTime(0, useHours)} 確實就是投影片畫面。
    * 若畫面中只有講者、觀眾或過場動畫，請**忽略**該影格，不要強行匹配。

2.  **${pdf2Images.length > 0 ? '雙份簡報切換邏輯 (PDF 1 -> PDF 2)' : '單份簡報分析邏輯'}**：
    * 影片內容是連續的。順序必然是：先展示 PDF 1 的頁面${pdf2Images.length > 0 ? ' -> (可能有一段講者串場/休息) -> 接著展示 PDF 2 的頁面' : ''}。
    ${pdf2Images.length > 0 ?
      '* 如果提供了 PDF 2，則依序偵測從 PDF 1 切換到 PDF 2 的時間點。' :
      '* 如果沒有提供 PDF 2，請僅針對 PDF 1 進行分析，忽略任何不屬於 PDF 1 的內容。'}
    * ${pdf2Images.length > 0 ? 'PDF 1 與 PDF 2 之間**只會發生一次**切換。' : ''}
    * 在切換期間（例如換檔空檔），若畫面無投影片，請勿產生匹配事件。

3.  **視覺匹配優先**：
    * 請根據畫面中的文字標題、圖表形狀、圖片排版進行比對。
    * **標題識別**：請優先讀取投影片上方的大字體標題作為 \`slideTitle\`。若無標題，請總結畫面核心內容。

4.  **輸出格式 (JSON)**：
    請輸出一個 JSON 物件，包含一個 \`transitions\` 陣列。每個元素代表一次「投影片更換事件」。
    格式要求：
    * \`timestamp\`: 字串 (${timeFormatLabel})，該投影片**首次**出現的影格時間，請直接使用該影格的 VIDEO_TIMESTAMP 值。
    * \`pdfId\`: 整數 (${pdf2Images.length > 0 ? '1 或 2' : '固定為 1'})，代表屬於哪一份 PDF。
    * \`pageNumber\`: 整數，對應的 PDF 頁碼。
    * \`slideTitle\`: 字串，投影片標題。
    * \`reasoning\`: 字串 (正體中文)，簡述判斷理由 (例如：「畫面標題與 PDF 1 第 3 頁一致」、「圖表吻合」)。請保持簡短，以避免輸出長度超出 token 上限。
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
            transitions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  timestamp: { type: Type.STRING },
                  pdfId: { type: Type.INTEGER },
                  pageNumber: { type: Type.INTEGER },
                  slideTitle: { type: Type.STRING },
                  reasoning: { type: Type.STRING },
                  confidence: { type: Type.STRING, enum: ["High", "Medium", "Low"] }
                },
                required: ["timestamp", "pdfId", "pageNumber", "slideTitle", "reasoning", "confidence"]
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
      throw new UserFacingError(`模型回應為空${detail}。可能是內容被安全過濾器攔截，或輸出在產生前就已用盡 token，請稍後再試或縮短影片。`);
    }

    const { transitions: rawTransitions, truncated } = parseTransitionsJson(responseText);

    // --- Midpoint Correction ---
    // A slide first seen at frame t actually appeared somewhere in (t - interval, t],
    // so shift back by half the sampling interval (never below 0).
    // Prefer the exact sampled time of the frame whose label the model returned.
    const frameTimes = new Map<string, number>();
    videoFrames.forEach(f => {
      if (!frameTimes.has(f.timestamp)) frameTimes.set(f.timestamp, f.time);
    });
    const correction = interval / 2;

    const transitions: SlideMatch[] = rawTransitions.map((t: any) => {
      const rawTimestamp = String(t.timestamp ?? '').trim();
      const detectedSeconds = frameTimes.get(rawTimestamp) ?? parseTimeToSeconds(rawTimestamp);
      if (!Number.isFinite(detectedSeconds)) {
        console.warn("Unparseable timestamp from model:", t.timestamp);
        return { ...t, timestamp: rawTimestamp, seconds: NaN };
      }
      const correctedSeconds = Math.max(0, detectedSeconds - correction);
      return {
        ...t,
        seconds: correctedSeconds,
        timestamp: formatTime(correctedSeconds, useHours),
      };
    });

    return { transitions, truncated };
  };

  // --- Main Handler ---
  const processFiles = async () => {
    // UI Validation Change: Only require Video and PDF 1
    if (!videoFile || !pdf1File) {
      alert("請至少上傳影片與 PDF 1（主要簡報）。");
      return;
    }

    setIsAnalyzing(true);
    setStatus('正在平行處理檔案（影片與 PDF）...');
    setError(null);
    setResults([]);

    try {
      // 1. Performance Optimization: Parallel Processing
      // Extraction Logic Change: Conditionally extract PDF 2 images
      const pdf1Promise = extractPdfImages(pdf1File);
      const pdf2Promise = pdf2File ? extractPdfImages(pdf2File) : Promise.resolve([]);
      const videoPromise = extractVideoFrames(videoFile);

      const [pdf1Images, pdf2Images, video] = await Promise.all([
        pdf1Promise,
        pdf2Promise,
        videoPromise
      ]);

      if (video.frames.length === 0) {
        throw new UserFacingError("影格抽取失敗：無法從影片取得任何畫面。請確認影片格式為瀏覽器可播放的 MP4/WebM，且檔案未損壞。");
      }
      if (pdf1Images.length === 0) {
        throw new UserFacingError("PDF 1 轉換失敗：無法取得任何頁面，請確認檔案未損壞。");
      }

      setStatus(`已抽取：PDF 1 共 ${pdf1Images.length} 頁、PDF 2 共 ${pdf2Images.length} 頁、影片 ${video.frames.length} 張影格。正在以 Gemini 分析...`);

      // 2. Gemini Analysis
      const { transitions, truncated } = await analyzeWithGemini(pdf1Images, pdf2Images, video);

      setResults(transitions);
      if (transitions.length === 0) {
        setStatus(truncated ? '模型回應被截斷，且沒有任何完整的結果。' : '未偵測到任何投影片切換。');
      } else if (truncated) {
        setStatus(`分析完成，但模型回應被截斷：已保留前 ${transitions.length} 筆完整結果，最後一筆不完整的資料已捨棄。`);
      } else {
        setStatus(`分析完成，共 ${transitions.length} 筆結果。`);
      }

    } catch (err) {
      console.error(err);
      setStatus('');
      setError(toErrorMessage(err));
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto font-sans">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">影片投影片分析工具（單份或雙份簡報）</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
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
        <div className="bg-white rounded-xl shadow-md overflow-hidden border border-gray-200">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-100 text-gray-700 text-sm uppercase tracking-wider">
                <th className="p-4 border-b">時間</th>
                <th className="p-4 border-b">來源</th>
                <th className="p-4 border-b">頁碼</th>
                <th className="p-4 border-b">標題／摘要</th>
                <th className="p-4 border-b">信心度</th>
                <th className="p-4 border-b">判斷理由</th>
              </tr>
            </thead>
            <tbody>
              {results.map((match, index) => (
                <tr key={index} className="hover:bg-gray-50 border-b last:border-0 transition-colors">
                  <td className="p-4">
                    <button
                      onClick={() => jumpToTime(match.seconds)}
                      disabled={!Number.isFinite(match.seconds)}
                      title="跳到影片此時間點並播放"
                      className="inline-flex items-center gap-1 px-3 py-1 rounded-lg font-mono font-bold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 disabled:text-gray-400 disabled:bg-gray-100 disabled:cursor-not-allowed transition-colors"
                    >
                      <span aria-hidden="true">▶</span>
                      {match.timestamp}
                    </button>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
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
