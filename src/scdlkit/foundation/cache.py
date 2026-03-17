"""Checkpoint caching helpers for foundation-model integrations."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from platformdirs import user_cache_dir

try:
    import gdown
except ImportError:  # pragma: no cover - exercised in minimal-install CI
    gdown = None

DEFAULT_SCGPT_CHECKPOINT = "whole-human"
REQUIRED_CHECKPOINT_FILES = ("args.json", "best_model.pt", "vocab.json")

_SCGPT_CHECKPOINTS: dict[str, dict[str, str]] = {
    "whole-human": {
        "description": "Pretrained on 33 million normal human cells.",
        "folder_id": "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
        "url": "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
    }
}


def list_scgpt_checkpoints() -> dict[str, dict[str, str]]:
    """Return the supported official scGPT checkpoints."""

    return {
        checkpoint_id: checkpoint.copy()
        for checkpoint_id, checkpoint in _SCGPT_CHECKPOINTS.items()
    }


def get_cache_root(cache_dir: str | Path | None = None) -> Path:
    """Resolve the cache root for scDLKit foundation assets."""

    if cache_dir is not None:
        root = Path(cache_dir)
    else:
        env_cache_dir = os.environ.get("SCDLKIT_CACHE_DIR")
        root = Path(env_cache_dir) if env_cache_dir else Path(user_cache_dir("scdlkit"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_scgpt_checkpoint_dir(
    checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the expected local path for a cached scGPT checkpoint."""

    if checkpoint not in _SCGPT_CHECKPOINTS:
        msg = f"Unsupported scGPT checkpoint '{checkpoint}'."
        raise ValueError(msg)
    return get_cache_root(cache_dir) / "foundation" / "scgpt" / checkpoint


def _has_required_checkpoint_files(path: Path) -> bool:
    return all((path / filename).exists() for filename in REQUIRED_CHECKPOINT_FILES)


def _resolve_download_root(path: Path) -> Path:
    if _has_required_checkpoint_files(path):
        return path
    child_directories = [child for child in path.iterdir() if child.is_dir()]
    for child in child_directories:
        if _has_required_checkpoint_files(child):
            return child
    msg = f"Downloaded checkpoint under '{path}' is missing the expected files."
    raise RuntimeError(msg)


def ensure_scgpt_checkpoint(
    checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download the official scGPT checkpoint into the local cache if needed."""

    target_dir = get_scgpt_checkpoint_dir(checkpoint, cache_dir=cache_dir)
    if target_dir.exists() and _has_required_checkpoint_files(target_dir) and not force_download:
        return target_dir

    checkpoint_info = _SCGPT_CHECKPOINTS.get(checkpoint)
    if checkpoint_info is None:
        msg = f"Unsupported scGPT checkpoint '{checkpoint}'."
        raise ValueError(msg)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)

    download_dir = target_dir.parent / f".{target_dir.name}.download"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    if gdown is None:
        msg = (
            "gdown is required to download scGPT checkpoints. "
            "Install scdlkit with the foundation extra: `pip install \"scdlkit[foundation]\"`."
        )
        raise ImportError(msg)

    downloaded_files = gdown.download_folder(
        id=checkpoint_info["folder_id"],
        output=str(download_dir),
        quiet=False,
        remaining_ok=True,
        resume=True,
    )
    if not downloaded_files:
        msg = f"Failed to download scGPT checkpoint '{checkpoint}'."
        raise RuntimeError(msg)

    resolved_root = _resolve_download_root(download_dir)
    shutil.move(str(resolved_root), str(target_dir))
    if download_dir.exists():
        shutil.rmtree(download_dir)
    return target_dir
