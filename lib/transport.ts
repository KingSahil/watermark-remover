const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000/api"
).replace(/\/+$/, "");

const UPLOAD_SHARE = 0.4;
const REQUEST_TIMEOUT_MS = 600_000;

export interface ProcessResult {
  blob: Blob;
  width: number;
  height: number;
  maskBlob: Blob | null;
}

export interface ProcessHandlers {
  onProgress: (fraction: number) => void;
  onStage?: (label: string) => void;
  onMask?: (maskUrl: string) => void;
}

interface StreamMessage {
  stage?: string;
  label?: string;
  progress?: number;
  error?: string;
  image?: string;
  mask?: string;
  width?: number;
  height?: number;
  detail?: string;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function base64ToBlob(encoded: string, type: string): Blob {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type });
}

function decodeImage(message: StreamMessage, maskBlob: Blob | null): ProcessResult {
  const encoded = message.image;
  if (typeof encoded !== "string" || encoded.length === 0) {
    throw new Error("The server returned an empty image.");
  }
  return {
    blob: base64ToBlob(encoded, "image/png"),
    width: message.width ?? 0,
    height: message.height ?? 0,
    maskBlob,
  };
}

function readHttpError(xhr: XMLHttpRequest): string {
  try {
    const parsed = JSON.parse(xhr.responseText) as StreamMessage;
    if (typeof parsed.detail === "string" && parsed.detail.length > 0) {
      return parsed.detail;
    }
  } catch {
    // Fall through to the generic message below.
  }
  return `The server rejected the image (HTTP ${xhr.status}).`;
}

/**
 * Streams the upload and the backend's NDJSON processing stages through one
 * XHR: upload events report real transfer progress, response events report
 * the backend's stage progress and human-readable stage labels, and the
 * water mask arrives as its own event (plus again with `done`) so the UI can
 * animate evaporation inside the masked region while diffusion runs.
 */
export function processImage(
  file: File,
  handlers: ProcessHandlers,
): Promise<ProcessResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/process`);
    xhr.timeout = REQUEST_TIMEOUT_MS;

    let buffer = "";
    let cursor = 0;
    let result: ProcessResult | null = null;
    let streamError: string | null = null;
    let maskBlob: Blob | null = null;

    const handleLine = (line: string) => {
      let message: StreamMessage;
      try {
        message = JSON.parse(line) as StreamMessage;
      } catch {
        return;
      }
      if (typeof message.error === "string") {
        streamError = message.error;
        return;
      }
      if (typeof message.mask === "string" && message.mask.length > 0) {
        maskBlob = base64ToBlob(message.mask, "image/png");
        handlers.onMask?.(URL.createObjectURL(maskBlob));
      }
      if (typeof message.image === "string") {
        result = decodeImage(message, maskBlob);
        return;
      }
      const label =
        typeof message.label === "string" && message.label.length > 0
          ? message.label
          : typeof message.stage === "string"
            ? message.stage
            : null;
      if (label) {
        handlers.onStage?.(label);
      }
      if (typeof message.progress === "number") {
        const serverShare =
          UPLOAD_SHARE + (1 - UPLOAD_SHARE) * clamp01(message.progress);
        handlers.onProgress(serverShare);
      }
    };

    const consume = (flush: boolean) => {
      const text = xhr.responseText;
      if (text.length > cursor) {
        buffer += text.slice(cursor);
        cursor = text.length;
      }
      let boundary = buffer.indexOf("\n");
      while (boundary >= 0) {
        const line = buffer.slice(0, boundary).trim();
        buffer = buffer.slice(boundary + 1);
        if (line.length > 0) {
          handleLine(line);
        }
        boundary = buffer.indexOf("\n");
      }
      if (flush && buffer.trim().length > 0) {
        handleLine(buffer.trim());
        buffer = "";
      }
    };

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        handlers.onProgress(UPLOAD_SHARE * (event.loaded / event.total));
      }
    };

    xhr.onprogress = () => consume(false);

    xhr.onload = () => {
      consume(true);
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(readHttpError(xhr)));
        return;
      }
      if (streamError) {
        reject(new Error(streamError));
        return;
      }
      if (result) {
        resolve(result);
        return;
      }
      reject(new Error("The server returned an unexpected response."));
    };

    xhr.onerror = () => {
      reject(
        new Error(
          `Could not reach the backend at ${API_BASE}. Start it with "uvicorn app.main:app".`,
        ),
      );
    };

    xhr.ontimeout = () => {
      reject(new Error("Processing timed out. Please try again."));
    };

    xhr.onabort = () => {
      reject(new Error("Processing was cancelled."));
    };

    const form = new FormData();
    form.append("file", file, file.name || "upload.png");
    xhr.send(form);
  });
}
