"""Device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve a user-facing device string into a torch device."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
