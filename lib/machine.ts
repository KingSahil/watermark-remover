import type { ProcessResult } from "./transport";

export type RemoverStep = "upload" | "processing" | "result" | "paywall";

export interface RemoverSnapshot {
  step: RemoverStep;
  progress: number;
  stage: string | null;
  previewUrl: string | null;
  finalUrl: string | null;
  maskUrl: string | null;
  error: string | null;
  payAttempted: boolean;
}

export type ProcessTransport = (
  file: File,
  handlers: {
    onProgress: (fraction: number) => void;
    onStage?: (label: string) => void;
    onMask?: (maskUrl: string) => void;
  },
) => Promise<ProcessResult>;

export const INITIAL_SNAPSHOT: RemoverSnapshot = {
  step: "upload",
  progress: 0,
  stage: null,
  previewUrl: null,
  finalUrl: null,
  maskUrl: null,
  error: null,
  payAttempted: false,
};

/** How long the evaporation finish (dissolve + canvas fade-out) lingers. */
const FINISH_MS = 1400;

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function messageOf(cause: unknown): string {
  if (cause instanceof Error && cause.message.trim().length > 0) {
    return cause.message;
  }
  return "Something went wrong while processing the image. Please try again.";
}

/**
 * Class-based step machine: owns every state transition of the
 * upload → processing → result → paywall flow. Components only read the
 * snapshot and call intent methods; the raw network function is injected
 * through the transport so the UI never touches XHR directly. While
 * processing, the snapshot carries the original preview, the streamed water
 * mask, and — once done — the final image, so ProcessingView can dissolve
 * original → final inside the mask before the result step takes over.
 */
export class WatermarkRemoverMachine {
  private snapshot: RemoverSnapshot = INITIAL_SNAPSHOT;
  private readonly listeners = new Set<() => void>();
  private requestSequence = 0;
  private previewUrl: string | null = null;
  private finalUrl: string | null = null;
  private maskUrl: string | null = null;
  private finishTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly transport: ProcessTransport) {}

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): RemoverSnapshot => this.snapshot;

  getServerSnapshot = (): RemoverSnapshot => INITIAL_SNAPSHOT;

  attach = (file: File): void => {
    if (this.snapshot.step === "processing") {
      return;
    }
    const requestId = ++this.requestSequence;
    this.clearFinish();
    this.releaseAllUrls();
    this.previewUrl = URL.createObjectURL(file);
    this.commit({
      step: "processing",
      progress: 0,
      stage: "Uploading…",
      previewUrl: this.previewUrl,
      finalUrl: null,
      maskUrl: null,
      error: null,
      payAttempted: false,
    });

    this.transport(file, {
      onProgress: (fraction) => {
        if (requestId !== this.requestSequence) {
          return;
        }
        const next = clamp01(fraction);
        if (next > this.snapshot.progress) {
          this.commit({ progress: next });
        }
      },
      onStage: (label) => {
        if (requestId !== this.requestSequence) {
          return;
        }
        this.commit({ stage: label });
      },
      onMask: (maskUrl) => {
        if (requestId !== this.requestSequence) {
          return;
        }
        this.releaseMask();
        this.maskUrl = maskUrl;
        this.commit({ maskUrl });
      },
    })
      .then((result) => {
        if (requestId !== this.requestSequence) {
          return;
        }
        // Finished: hold the processing view briefly so the canvas can
        // cross-dissolve original → final inside the mask, then swap to
        // the result step exactly as before.
        this.finalUrl = URL.createObjectURL(result.blob);
        this.commit({
          progress: 1,
          finalUrl: this.finalUrl,
        });
        this.finishTimer = setTimeout(() => {
          if (requestId !== this.requestSequence) {
            return;
          }
          this.releasePreview();
          this.releaseMask();
          this.previewUrl = this.finalUrl;
          this.finalUrl = null;
          this.commit({
            step: "result",
            progress: 1,
            previewUrl: this.previewUrl,
            finalUrl: null,
            maskUrl: null,
          });
        }, FINISH_MS);
      })
      .catch((cause: unknown) => {
        if (requestId !== this.requestSequence) {
          return;
        }
        this.clearFinish();
        this.releaseAllUrls();
        this.commit({
          step: "upload",
          progress: 0,
          stage: null,
          previewUrl: null,
          finalUrl: null,
          maskUrl: null,
          error: messageOf(cause),
        });
      });
  };

  requestDownload = (): void => {
    if (this.snapshot.step !== "result") {
      return;
    }
    this.commit({ step: "paywall", payAttempted: false });
  };

  acknowledgePaywall = (): void => {
    if (this.snapshot.step !== "paywall" || this.snapshot.payAttempted) {
      return;
    }
    this.commit({ payAttempted: true });
  };

  destroy = (): void => {
    this.requestSequence += 1;
    this.clearFinish();
    this.releaseAllUrls();
    this.listeners.clear();
  };

  private commit(patch: Partial<RemoverSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const notify of this.listeners) {
      notify();
    }
  }

  private clearFinish(): void {
    if (this.finishTimer !== null) {
      clearTimeout(this.finishTimer);
      this.finishTimer = null;
    }
  }

  private releasePreview(): void {
    if (this.previewUrl) {
      URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = null;
    }
  }

  private releaseMask(): void {
    if (this.maskUrl) {
      URL.revokeObjectURL(this.maskUrl);
      this.maskUrl = null;
    }
  }

  private releaseAllUrls(): void {
    this.releasePreview();
    this.releaseMask();
    if (this.finalUrl) {
      URL.revokeObjectURL(this.finalUrl);
      this.finalUrl = null;
    }
  }
}
