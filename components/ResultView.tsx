import { Download } from "lucide-react";

interface ResultViewProps {
  previewUrl: string | null;
  onDownload: () => void;
}

export function ResultView({ previewUrl, onDownload }: ResultViewProps) {
  if (!previewUrl) {
    return null;
  }

  return (
    <main className="flex min-h-dvh flex-col items-center justify-center gap-8 bg-canvas px-6 py-16">
      <figure className="animate-rise overflow-hidden rounded-md border border-hairline-soft bg-canvas">
        {/* eslint-disable-next-line @next/next/no-img-element -- blob: preview of unknown intrinsic size */}
        <img
          src={previewUrl}
          alt="Processed preview: the paper is crumpled and water-damaged while the watermark stays sharp"
          className="block max-h-[62vh] max-w-[min(88vw,880px)] object-contain"
        />
      </figure>

      <button
        type="button"
        onClick={onDownload}
        className="animate-rise inline-flex h-11 cursor-pointer items-center gap-2 rounded-full bg-ink px-6 text-base font-semibold text-canvas outline-none transition-[transform,opacity] duration-150 hover:opacity-90 active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-ink focus-visible:ring-offset-4 [animation-delay:90ms]"
      >
        <Download size={16} strokeWidth={2.25} aria-hidden="true" />
        Download
      </button>
    </main>
  );
}
