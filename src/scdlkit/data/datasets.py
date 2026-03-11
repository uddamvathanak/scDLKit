"""PyTorch datasets backed by dense or sparse matrices."""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

from scdlkit.data.schemas import SplitData


class AnnDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset that converts rows to dense float32 on access."""

    def __init__(self, split: SplitData):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.split.X[index]
        x = row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()
        sample: dict[str, torch.Tensor] = {"x": torch.as_tensor(x, dtype=torch.float32)}
        if self.split.labels is not None:
            sample["y"] = torch.as_tensor(int(self.split.labels[index]), dtype=torch.long)
        if self.split.batches is not None:
            sample["batch"] = torch.as_tensor(int(self.split.batches[index]), dtype=torch.long)
        return sample
