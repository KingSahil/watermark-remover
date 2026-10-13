"use client";

import { useEffect, useRef, useState, type DragEvent, type ChangeEvent } from "react";
import { ImagePlus } from "lucide-react";

interface UploadViewProps {
  error: string | null;
  onStart: (file: File) => void;
}

export function UploadView({ error, onStart }: UploadViewProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      const file = event.clipboardData?.files.item(0);
      if (file) {
        event.preventDefault();
        onStart(file);
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [onStart]);

  const handleDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files.item(0);
    if (file) {
      onStart(file);
    }
  };

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.item(0);
    event.target.value = "";
    if (file) {
      onStart(file);
    }
  };

  return (
    <div className="flex min-h-dvh flex-col items-center bg-canvas">
      <header className="pt-[72px] sm:pt-[96px]">
        <h1 className="text-center font-display text-[clamp(30px,5vw,44px)] font-light leading-none text-ink">
          Remove Watermark
        </h1>
      </header>

      <main className="w-full px-6 pt-11 sm:pt-14">
        <button
          type="button"
          aria-label="Upload an image"
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            if (event.dataTransfer) {
              event.dataTransfer.dropEffect = "copy";
            }
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={`mx-auto flex h-[clamp(220px,34vh,340px)] w-[min(560px,100%)] cursor-pointer flex-col items-center justify-center gap-3 rounded-md border-2 border-dotted outline-none transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-ink focus-visible:ring-offset-4 ${
            dragging
              ? "border-ink bg-canvas-soft"
              : "border-faint bg-canvas hover:border-muted"
          }`}
        >
          <ImagePlus
            aria-hidden="true"
            className={`h-8 w-8 transition-colors duration-200 ${
              dragging ? "text-ink" : "text-faint"
            }`}
            strokeWidth={1.5}
          />
          <span className="text-sm text-muted">
            {dragging ? "Drop it like it's wet" : "Drag image here"}
          </span>
          <span className="-mt-2 text-xs text-faint">
            or click to browse, or paste
          </span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          tabIndex={-1}
          className="sr-only"
          onChange={handleChange}
        />
        {error ? (
          <p
            role="alert"
            className="mx-auto mt-5 w-[min(560px,100%)] text-center text-sm text-muted"
          >
            {error}
          </p>
        ) : null}
      </main>
    </div>
  );
}
