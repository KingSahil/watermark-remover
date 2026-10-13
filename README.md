# Remove Watermark

A hyper-minimal single-page utility that takes "Watermark Remover" completely literally.
Instead of erasing text, it **removes the water** — CLIPSeg finds whatever liquid is in
the photo, Stable Diffusion evaporates it into a dry, cracked lakebed, and the original
`WATERMARK` stays perfectly sharp on top. Then a $20.00 paywall traps you.

Gallery-white Mobbin-style UI (see `DESIGN.md`), a class-based four-state machine on the
frontend, and a GPU AI pipeline (CLIPSeg segmentation + SD inpainting) on the backend.
While the model works you watch the water boil off live: steam particles rise out of the
masked region and the surface dries and dissolves in sync with real progress.

## Stack

- **Frontend:** Next.js 16 (App Router), Tailwind CSS v4, Lucide React, TypeScript
- **Backend:** FastAPI, PyTorch (CUDA 12.8), Hugging Face `transformers` (CLIPSeg),
  `diffusers` (Stable Diffusion inpainting), OpenCV, NumPy

## Quick start

```bash
# 1. Backend (terminal A)
cd backend
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt     # macOS/Linux: .venv/bin/pip
.venv\Scripts\python scripts\prefetch_models.py   # one-time ~4.5 GB model download
.venv\Scripts\uvicorn app.main:app                # macOS/Linux: .venv/bin/uvicorn

# 2. Frontend (terminal B)
npm install
npm run dev
```

Open http://localhost:3000, drag an image onto the dotted dropzone, watch the bar fill
(`Detecting water…` → `Evaporating…` → `Keeping watermark…`) while steam rises off the
water, then fall into the $20.00 trap on Download.

## GPU setup

Developed against an NVIDIA RTX 4070 Laptop (8 GB VRAM). The pipeline deliberately
stays in SD 1.5 territory — SDXL does not fit in 8 GB.

- `torch` must be the CUDA build: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  (cp314 Windows wheels exist for torch ≥ 2.11).
- The SD inpainting pipeline runs in `float16` on CUDA with **attention slicing** and
  **VAE slicing** enabled for 8 GB safety.
- Models load lazily on the first request and are cached in the default Hugging Face
  cache (`%USERPROFILE%\.cache\huggingface`); GPU inference is serialized behind a lock.
- CLIPSeg runs on CUDA when available and falls back to CPU otherwise.
- On CUDA OOM or model-load failure the backend streams a clean
  `{"stage":"error","error":"…"}` event — it never silently ships a bad mask.

## The four states

| State | Surface | Transition |
| --- | --- | --- |
| `upload` | Light geometric header + dotted dropzone with drag prompt | file dropped → |
| `processing` | Original-image preview + evaporation canvas + 2px bar + live stage label | transport resolves → |
| `result` | Centered preview + ink Download pill | Download clicked → |
| `paywall` | White slide-up card over the still-visible preview | never (that's the joke) |

Every transition lives in `WatermarkRemoverMachine` (`lib/machine.ts`); components only
read the snapshot and call intent methods (`attach`, `requestDownload`, `acknowledgePaywall`).
The raw XHR/NDJSON transport is injected, so the UI never touches the network directly.
During the final ~1.4 s of `processing` the snapshot also carries the finished image, so
`ProcessingView` can cross-dissolve original → result inside the water mask before the
`result` step takes over.

## On-screen evaporation

The backend streams the downscaled water mask (~256 px PNG, base64) as its own NDJSON
event as soon as segmentation finishes (and again with `done`). `ProcessingView` paints a
`<canvas>` exactly over the preview image and, driven only by real `snapshot.progress`:

- progressively dries the masked region (desaturated/warmed stand-in, then the true
  SD result cross-dissolves in once it arrives),
- spawns rising steam/vapor particles at mask pixels, density ∝ progress,
- fades the canvas out at progress = 1, right before the machine transitions to `result`.

`prefers-reduced-motion` skips the particles and keeps a plain dissolve. No timers pace
any of it — the backend's per-diffusion-step events are the clock.

## Progress with narration

`lib/transport.ts` drives one XHR: upload events fill the first 40% of the bar, then each
NDJSON line advances it the rest of the way. Segmentation covers ~0–15%, every diffusion
step moves the bar via diffusers' `callback_on_step_end` (~15→90%), and watermark
protection takes it to 100%. Labels shown under the bar: `Detecting water…`,
`Evaporating…`, `Keeping watermark…`.

## API

`POST /api/process` — multipart field `file` → `application/x-ndjson` stream:

```jsonl
{"stage":"detect","label":"Detecting water…","progress":0.02}
{"stage":"mask","label":"Detecting water…","progress":0.15,"mask":"<base64 PNG ~256px>"}
{"stage":"evaporate","label":"Evaporating…","progress":0.19}
{"stage":"evaporate","label":"Evaporating…","progress":0.23}
… (one event per diffusion step, ~20 total)
{"stage":"protect","label":"Keeping watermark…","progress":0.92}
{"stage":"done","progress":1.0,"image":"<base64 PNG>","width":1200,"height":800,"mask":"<base64 PNG ~256px>"}
```

The `mask` is a grayscale PNG at ~256 px width; white = water (the region being
evaporated). Errors: `413` over 15 MB, `422` undecodable image, mid-stream
`{"stage":"error","error":"…"}`. `GET /api/health` returns `{"status":"ok"}`.

## Pipeline (`backend/app/processing/`)

1. `ai_models.py` — lazy, process-wide model registry; CUDA detection; serialized GPU
   inference; clean `WaterPipelineError` on load failure/OOM.
2. `segment.py` — CLIPSeg (`CIDAS/clipseg-rd64-refined`) prompted
   `["water","lake","river","sea","pool"]`; per-prompt probabilities are unioned,
   thresholded (with adaptive fallbacks), cleaned with morphological open/close, and
   filtered to the largest components. Works on any photo with water in it. If nothing
   sensible is found, a feathered lower-half mask keeps the joke alive. The AI mask is
   then unioned with chroma-turquoise pixels touching it (catches the far-shore strip
   CLIPSeg scores low), closed wide to fill speckle holes, and dilated — over-masking
   the shoreline blends seamlessly, under-masking leaves turquoise rims behind.
3. `inpaint.py` — `runwayml/stable-diffusion-inpainting` (fp16, 512 px longest side,
   ~20 steps, attention + VAE slicing) repaints the water as *"dry cracked earth
   lakebed, sun-baked mud"*; each step fires a callback; the result is composited back
   into the full-resolution original with a feathered shoreline. Both the SD input mask
   and the composite alpha are grown (never eroded). The negative prompt also bans
   `teal, cyan, reflection, puddle, sign, billboard, text` so diffusion neither
   repaints water around boats nor hallucinates signage. After watermark protection, a
   `sweep_residual_water` pass drains any leftover strict-turquoise (SD bleed, glyph
   counters, shoreline confetti) toward gray: big blobs must touch the mask zone, small
   ones die unless sky-connected through an erosion-severed map — the sky itself is
   never touched.
4. `text.py` — OCR-free text-line detection; the kept blobs are intersected with a
   contrast overlay AND a glyph-tone gate (near-white/near-black only), so boats and
   wakes fused into the same blob are not protected. Detected glyphs are layered back
   **sharp from the original** after inpainting. Uploads with no text get the literal
   `WATERMARK` stamped on top instead — every output keeps its watermark.
5. `processor.py` — orchestrates detect → segment → diffuse → protect → encode, emits
   labeled NDJSON progress (an async generator; heavy work in threadpools/worker
   threads, zero pacing sleeps), and encodes both the full PNG and the small mask.

Seeded from `crc32(upload)`, so the same file always evaporates the same way.

## Configuration

| Variable | Default | Where |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE` | `http://127.0.0.1:8000/api` | frontend |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | backend |

## Scripts

```bash
npm run lint            # eslint
npm run build           # production build (type-checks)
```

```bash
cd backend
.venv\Scripts\python.exe scripts\make_sample.py        # generate test inputs
.venv\Scripts\python.exe scripts\prefetch_models.py    # one-time model download
.venv\Scripts\python.exe scripts\smoke_test.py         # full-pipeline assertions (GPU)
.venv\Scripts\python.exe scripts\test_endpoint.py      # NDJSON stream over HTTP (server running)
.venv\Scripts\python.exe scripts\test_error_path.py    # 422 path
```
