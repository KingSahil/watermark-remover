from __future__ import annotations

import json
import logging
import zlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .config import settings
from .errors import InvalidImageError
from .processing import WatermarkProcessor

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Watermark Remover",
    version="2.0.0",
    description="AI-evaporates the water out of photos (CLIPSeg + Stable Diffusion) "
    "while keeping the watermark razor sharp.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

processor = WatermarkProcessor()


async def ndjson(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    """Serialize stage events as newline-delimited JSON lines."""
    async for event in events:
        yield json.dumps(event, separators=(",", ":")) + "\n"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process")
async def process(file: UploadFile = File(...)) -> StreamingResponse:
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. The limit is {limit_mb} MB.",
        )

    try:
        image = await run_in_threadpool(processor.decode, raw)
    except InvalidImageError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    seed = zlib.crc32(raw) & 0xFFFFFFFF
    return StreamingResponse(
        ndjson(processor.stream(image, seed)),
        media_type="application/x-ndjson",
    )
