"use client";

import { useEffect, useRef, useState } from "react";

interface ProcessingViewProps {
  progress: number;
  stage: string | null;
  previewUrl: string | null;
  finalUrl: string | null;
  maskUrl: string | null;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  age: number;
  life: number;
  size: number;
}

const DISSOLVE_START = 0.15;
const FINISH_DISSOLVE_MS = 1_100;
const MAX_PARTICLES = 220;
const SPAWN_PER_SECOND = 120;

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function smoothstep(edge: number): number {
  const x = clamp01(edge);
  return x * x * (3 - 2 * x);
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${src.slice(0, 32)}`));
    image.src = src;
  });
}

function makeSprite(): HTMLCanvasElement {
  const sprite = document.createElement("canvas");
  sprite.width = 64;
  sprite.height = 64;
  const ctx = sprite.getContext("2d");
  if (ctx) {
    const glow = ctx.createRadialGradient(32, 32, 2, 32, 32, 32);
    glow.addColorStop(0, "rgba(255,255,255,0.9)");
    glow.addColorStop(0.35, "rgba(255,255,255,0.35)");
    glow.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, 64, 64);
  }
  return sprite;
}

/**
 * Desaturates, warms, and lightens the water region of the original — the
 * client-side stand-in for the lakebed the backend is inpainting, so the
 * surface visibly dries while diffusion runs.
 */
function buildDriedLayer(
  original: HTMLImageElement,
  mask: HTMLImageElement,
): HTMLCanvasElement | null {
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const layer = document.createElement("canvas");
  layer.width = width;
  layer.height = height;
  const ctx = layer.getContext("2d", { willReadFrequently: true });
  const maskCtx = document
    .createElement("canvas")
    .getContext("2d", { willReadFrequently: true });
  if (!ctx || !maskCtx) {
    return null;
  }
  ctx.drawImage(original, 0, 0);
  maskCtx.canvas.width = width;
  maskCtx.canvas.height = height;
  maskCtx.drawImage(mask, 0, 0, width, height);

  const pixels = ctx.getImageData(0, 0, width, height);
  const water = maskCtx.getImageData(0, 0, width, height).data;
  const data = pixels.data;
  for (let index = 0; index < data.length; index += 4) {
    if (water[index] < 128) {
      continue;
    }
    const gray =
      0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
    const lift = gray * 0.12;
    data[index] = Math.min(255, gray * 0.3 + data[index] * 0.7 + lift + 10);
    data[index + 1] = Math.min(255, gray * 0.3 + data[index + 1] * 0.7 + lift + 4);
    data[index + 2] = Math.min(255, gray * 0.3 + data[index + 2] * 0.7 + lift - 6);
  }
  ctx.putImageData(pixels, 0, 0);
  return layer;
}

function sampleMaskPoints(mask: HTMLImageElement): Float32Array | null {
  const width = 128;
  const height = Math.max(
    1,
    Math.round((128 * mask.naturalHeight) / mask.naturalWidth),
  );
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    return null;
  }
  ctx.drawImage(mask, 0, 0, width, height);
  const data = ctx.getImageData(0, 0, width, height).data;
  const points: number[] = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (data[(y * width + x) * 4] > 128) {
        points.push((x + 0.5) / width, (y + 0.5) / height);
      }
    }
  }
  return points.length > 0 ? new Float32Array(points) : null;
}

export function ProcessingView({
  progress,
  stage,
  previewUrl,
  finalUrl,
  maskUrl,
}: ProcessingViewProps) {
  const figureRef = useRef<HTMLElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const progressRef = useRef(progress);
  const finalUrlRef = useRef(finalUrl);
  const [reducedMotion, setReducedMotion] = useState(false);
  // The base image starts as the original upload. The moment the finished
  // result decodes, it becomes the base too — so when the canvas overlay
  // fades out (or the view unmounts), the final image is revealed, never a
  // flash of the original water.
  const [baseSrc, setBaseSrc] = useState<string | null>(previewUrl);

  const percent = Math.round(clamp01(progress) * 100);

  useEffect(() => {
    progressRef.current = progress;
  }, [progress]);

  useEffect(() => {
    finalUrlRef.current = finalUrl;
  }, [finalUrl]);

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReducedMotion(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const figure = figureRef.current;
    if (!canvas || !figure || !previewUrl || !maskUrl) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    let disposed = false;
    let raf = 0;
    const particles: Particle[] = [];
    const sprite = makeSprite();

    let maskPoints: Float32Array | null = null;
    let maskImage: HTMLImageElement | null = null;
    let maskCanvas: HTMLCanvasElement | null = null;
    let driedLayer: HTMLCanvasElement | null = null;
    let finalImage: HTMLImageElement | null = null;
    let finalLoadingFor: string | null = null;
    let finalShownAt = 0;
    let spawnDebt = 0;
    let lastTime = performance.now();

    const rebuildMaskCanvas = () => {
      if (!maskImage) {
        maskCanvas = null;
        return;
      }
      const layer = document.createElement("canvas");
      layer.width = canvas.width;
      layer.height = canvas.height;
      const layerCtx = layer.getContext("2d");
      if (!layerCtx) {
        maskCanvas = null;
        return;
      }
      layerCtx.drawImage(maskImage, 0, 0, canvas.width, canvas.height);
      maskCanvas = layer;
    };

    const resize = () => {
      const rect = figure.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      rebuildMaskCanvas();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(figure);

    const frame = (now: number) => {
      if (disposed) {
        return;
      }
      const delta = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      const t = smoothstep(
        (clamp01(progressRef.current) - DISSOLVE_START) / (1 - DISSOLVE_START),
      );

      // Lazy-load the final image the moment the machine surfaces it.
      const wanted = finalUrlRef.current;
      if (wanted && !finalImage && finalLoadingFor !== wanted) {
        finalLoadingFor = wanted;
        loadImage(wanted)
          .then((image) => {
            if (disposed) {
              return;
            }
            finalImage = image;
            finalShownAt = performance.now();
            // Same URL the <img> below will use; already decoded, so the
            // swap below paints instantly with no blank frame.
            setBaseSrc(wanted);
          })
          .catch(() => {});
      }

      const ft = finalImage
        ? clamp01((now - finalShownAt) / FINISH_DISSOLVE_MS)
        : 0;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Dry-out approximation then the real result, clipped to the water mask.
      if (driedLayer && maskCanvas && (t > 0 || ft > 0)) {
        ctx.globalAlpha = t;
        ctx.drawImage(driedLayer, 0, 0, canvas.width, canvas.height);
        if (finalImage && ft > 0) {
          ctx.globalAlpha = ft;
          ctx.drawImage(finalImage, 0, 0, canvas.width, canvas.height);
        }
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = "destination-in";
        ctx.drawImage(maskCanvas, 0, 0);
        ctx.globalCompositeOperation = "source-over";
      }

      // Rising steam, spawned only at water pixels, density ∝ progress.
      if (!reducedMotion && maskPoints && t > 0) {
        spawnDebt += SPAWN_PER_SECOND * t * delta;
        while (spawnDebt >= 1 && particles.length < MAX_PARTICLES) {
          spawnDebt -= 1;
          const pick = Math.floor(Math.random() * (maskPoints.length / 2)) * 2;
          const scale = canvas.width / 640;
          particles.push({
            x: maskPoints[pick] * canvas.width,
            y: maskPoints[pick + 1] * canvas.height,
            vx: (Math.random() - 0.5) * 6 * scale,
            vy: -(28 + Math.random() * 42) * scale,
            age: 0,
            life: 1 + Math.random() * 0.9,
            size: (7 + Math.random() * 9) * scale,
          });
        }
        const fade = 1 - ft;
        for (let index = particles.length - 1; index >= 0; index -= 1) {
          const particle = particles[index];
          particle.age += delta;
          if (particle.age >= particle.life || fade <= 0) {
            particles.splice(index, 1);
            continue;
          }
          const phase = particle.age / particle.life;
          const wobble = Math.sin(now / 320 + index * 1.7) * 5 * (canvas.width / 640);
          const size = particle.size * (0.6 + phase * 1.6);
          particle.x += (particle.vx + wobble) * delta;
          particle.y += particle.vy * delta;
          ctx.globalAlpha = Math.sin(Math.PI * phase) * 0.24 * fade;
          ctx.drawImage(sprite, particle.x - size / 2, particle.y - size / 2, size, size);
        }
        ctx.globalAlpha = 1;
      }

      // Finish: fade the overlay out; the machine swaps to result.
      canvas.style.opacity = finalImage
        ? String(1 - smoothstep((ft - 0.7) / 0.3))
        : "1";

      raf = requestAnimationFrame(frame);
    };

    (async () => {
      try {
        const [original, mask] = await Promise.all([
          loadImage(previewUrl),
          loadImage(maskUrl),
        ]);
        if (disposed) {
          return;
        }
        maskImage = mask;
        maskPoints = sampleMaskPoints(mask);
        driedLayer = buildDriedLayer(original, mask);
        rebuildMaskCanvas();
        raf = requestAnimationFrame(frame);
      } catch {
        // The view still works; the overlay simply stays quiet.
      }
    })();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, [previewUrl, maskUrl, reducedMotion]);

  return (
    <div className="flex min-h-dvh animate-fade flex-col items-center justify-center gap-8 bg-canvas px-6 py-16">
      {baseSrc ? (
        <figure
          ref={figureRef}
          className="relative overflow-hidden rounded-md border border-hairline-soft bg-canvas"
        >
          {/* eslint-disable-next-line @next/next/no-img-element -- blob: preview of unknown intrinsic size */}
          <img
            src={baseSrc}
            alt="Image preview while the water evaporates"
            className="block max-h-[54vh] max-w-[min(88vw,880px)] object-contain"
          />
          <canvas
            ref={canvasRef}
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
        </figure>
      ) : null}
      <div className="flex flex-col items-center gap-5">
        <div
          role="progressbar"
          aria-label="Processing image"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          className="h-[2px] w-[min(320px,72vw)] overflow-hidden rounded-full bg-field"
        >
          <div
            className="h-full rounded-full bg-ink transition-[width] duration-150 ease-linear"
            style={{ width: `${percent}%` }}
          />
        </div>
        <p aria-live="polite" className="min-h-5 text-sm text-muted">
          {stage ?? " "}
        </p>
      </div>
    </div>
  );
}
