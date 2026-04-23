import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";
import * as pdfjsLib from 'pdfjs-dist';

// Initialize PDF.js worker
// Using esm.sh for the worker ensures version compatibility with the library imported from esm.sh
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://esm.sh/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

interface SlideMatch {
  timestamp: string;
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
  timestamp: string;
}

const App = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [pdf1File, setPdf1File] = useState<File | null>(null);
  const [pdf2File, setPdf2File] = useState<File | null>(null);
  const [status, setStatus] = useState<string>('');
  const [results, setResults] = useState<SlideMatch[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

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
  const extractVideoFrames = async (file: File): Promise<VideoFrame[]> => {
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
          resolve([]);
          return;
        }

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
                const mm = Math.floor(time / 60).toString().padStart(2, '0');
                const ss = Math.floor(time % 60).toString().padStart(2, '0');
                frames.push({
                  data: canvas.toDataURL('image/jpeg', 0.7).split(',')[1],
                  timestamp: `${mm}:${ss}`,
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
        resolve(frames);
      };

      // Handle general video load errors
      video.onerror = () => {
          console.error("Failed to load video metadata");
          URL.revokeObjectURL(video.src);
          resolve([]);
      };
    });
  };

  // --- Core: Analyze with Gemini ---
  const analyzeWithGemini = async (
    pdf1Images: PdfImage[], 
    pdf2Images: PdfImage[], 
    videoFrames: VideoFrame[]
  ) => {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    
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
    * **切勿強行從 00:00 開始**，除非 00:00 確實就是投影片畫面。
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
    * \`timestamp\`: 字串 (MM:SS)，該投影片**首次**出現的精確時間。
    * \`pdfId\`: 整數 (${pdf2Images.length > 0 ? '1 或 2' : '固定為 1'})，代表屬於哪一份 PDF。
    * \`pageNumber\`: 整數，對應的 PDF 頁碼。
    * \`slideTitle\`: 字串，投影片標題。
    * \`reasoning\`: 字串 (繁體中文)，簡述判斷理由 (例如：「畫面標題與 PDF 1 第 3 頁一致」、「圖表吻合」)。
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

    return JSON.parse(response.text || '{}');
  };

  // --- Main Handler ---
  const processFiles = async () => {
    // UI Validation Change: Only require Video and PDF 1
    if (!videoFile || !pdf1File) {
      alert("Please upload at least the Video and PDF 1 (Main).");
      return;
    }

    setIsAnalyzing(true);
    setStatus('Processing files (Parallel Mode)...');
    setResults([]);

    try {
      // 1. Performance Optimization: Parallel Processing
      // Extraction Logic Change: Conditionally extract PDF 2 images
      const pdf1Promise = extractPdfImages(pdf1File);
      const pdf2Promise = pdf2File ? extractPdfImages(pdf2File) : Promise.resolve([]);
      const videoPromise = extractVideoFrames(videoFile);

      const [pdf1Images, pdf2Images, videoFrames] = await Promise.all([
        pdf1Promise,
        pdf2Promise,
        videoPromise
      ]);

      setStatus(`Extracted: ${pdf1Images.length} pages (PDF1), ${pdf2Images.length} pages (PDF2), ${videoFrames.length} frames (Video). Analyzing with Gemini...`);

      // 2. Gemini Analysis
      const data = await analyzeWithGemini(pdf1Images, pdf2Images, videoFrames);
      
      if (data.transitions) {
        setResults(data.transitions);
        setStatus('Analysis Complete.');
      } else {
        setStatus('No transitions found.');
      }

    } catch (error) {
      console.error(error);
      setStatus('Error during analysis: ' + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto font-sans">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">Video Slide Analyzer (Single or Dual Deck)</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
          <label className="block text-sm font-semibold text-blue-900 mb-2">1. Video File (MP4/WebM)</label>
          <input 
            type="file" 
            accept="video/*" 
            onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
            className="w-full text-sm text-gray-700"
          />
        </div>

        <div className="p-4 bg-red-50 rounded-lg border border-red-100">
          <label className="block text-sm font-semibold text-red-900 mb-2">PDF 1 (Main)</label>
          <input 
            type="file" 
            accept="application/pdf" 
            onChange={(e) => setPdf1File(e.target.files?.[0] || null)}
            className="w-full text-sm text-gray-700"
          />
        </div>

        <div className="p-4 bg-green-50 rounded-lg border border-green-100">
          <label className="block text-sm font-semibold text-green-900 mb-2">PDF 2 (Optional)</label>
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
          {isAnalyzing ? 'Analyzing...' : 'Start Analysis'}
        </button>
        <p className="mt-4 text-gray-600 font-medium">{status}</p>
      </div>

      {results.length > 0 && (
        <div className="bg-white rounded-xl shadow-md overflow-hidden border border-gray-200">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-100 text-gray-700 text-sm uppercase tracking-wider">
                <th className="p-4 border-b">Time</th>
                <th className="p-4 border-b">Source</th>
                <th className="p-4 border-b">Page</th>
                <th className="p-4 border-b">Title / Summary</th>
                <th className="p-4 border-b">Confidence</th>
                <th className="p-4 border-b">Reasoning</th>
              </tr>
            </thead>
            <tbody>
              {results.map((match, index) => (
                <tr key={index} className="hover:bg-gray-50 border-b last:border-0 transition-colors">
                  <td className="p-4 font-mono font-bold text-indigo-700">{match.timestamp}</td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      match.pdfId === 1 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                    }`}>
                      PDF {match.pdfId}
                    </span>
                  </td>
                  <td className="p-4 font-medium">{match.pageNumber}</td>
                  <td className="p-4 font-semibold text-gray-800">{match.slideTitle}</td>
                  <td className="p-4">
                    <span className={`inline-block w-3 h-3 rounded-full mr-2 ${
                      match.confidence === 'High' ? 'bg-green-500' : 
                      match.confidence === 'Medium' ? 'bg-yellow-500' : 'bg-red-500'
                    }`}></span>
                    {match.confidence}
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