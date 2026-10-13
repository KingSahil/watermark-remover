from __future__ import annotations

import os
from dataclasses import dataclass, field


def _split_csv(name: str, default: str) -> tuple[str, ...]:
    values = os.getenv(name, default)
    return tuple(item.strip() for item in values.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    allowed_origins: tuple[str, ...] = field(
        default_factory=lambda: _split_csv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        )
    )
    max_upload_bytes: int = 15 * 1024 * 1024
    max_side: int = 3200
    png_compression: int = 3


settings = Settings()
