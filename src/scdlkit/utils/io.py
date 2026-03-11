"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
