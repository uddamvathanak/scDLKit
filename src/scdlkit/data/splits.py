"""Split helpers for prepared AnnData workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(slots=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _safe_train_test_split(
    indices: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    stratify: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        if stratify is None:
            raise
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def build_splits(
    n_samples: int,
    *,
    val_size: float,
    test_size: float,
    random_state: int,
    stratify: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> SplitIndices:
    """Create train/validation/test indices."""

    all_indices = np.arange(n_samples)
    if val_size < 0 or test_size < 0 or val_size + test_size >= 1:
        msg = "val_size and test_size must be >= 0 and sum to less than 1"
        raise ValueError(msg)

    holdout_fraction = val_size + test_size
    if holdout_fraction == 0:
        return SplitIndices(
            train=all_indices,
            val=np.array([], dtype=int),
            test=np.array([], dtype=int),
        )

    if groups is not None:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=holdout_fraction,
            random_state=random_state,
        )
        train_idx, holdout_idx = next(splitter.split(all_indices, groups=groups))
    else:
        train_idx, holdout_idx = _safe_train_test_split(
            all_indices,
            test_size=holdout_fraction,
            random_state=random_state,
            stratify=stratify,
        )

    if test_size == 0:
        return SplitIndices(
            train=np.sort(train_idx),
            val=np.sort(holdout_idx),
            test=np.array([], dtype=int),
        )

    holdout_stratify = stratify[holdout_idx] if stratify is not None else None
    if val_size == 0:
        return SplitIndices(
            train=np.sort(train_idx),
            val=np.array([], dtype=int),
            test=np.sort(holdout_idx),
        )

    test_fraction = test_size / holdout_fraction
    if groups is not None:
        holdout_groups = groups[holdout_idx]
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=random_state,
        )
        val_rel, test_rel = next(splitter.split(holdout_idx, groups=holdout_groups))
    else:
        val_rel, test_rel = _safe_train_test_split(
            np.arange(holdout_idx.size),
            test_size=test_fraction,
            random_state=random_state,
            stratify=holdout_stratify,
        )
    val_idx = holdout_idx[val_rel]
    test_idx = holdout_idx[test_rel]
    return SplitIndices(train=np.sort(train_idx), val=np.sort(val_idx), test=np.sort(test_idx))
