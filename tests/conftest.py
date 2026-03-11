from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

matplotlib.use("Agg")


def _make_obs(n_cells: int) -> pd.DataFrame:
    labels = np.array(["T", "B"] * (n_cells // 2))
    batches = np.array(
        ["batch1"] * (n_cells // 4)
        + ["batch2"] * (n_cells // 4)
        + ["batch3"] * (n_cells // 4)
        + ["batch4"] * (n_cells // 4)
    )
    index = [f"cell_{idx}" for idx in range(n_cells)]
    return pd.DataFrame({"cell_type": labels, "batch": batches}, index=index)


@pytest.fixture
def dense_adata() -> AnnData:
    rng = np.random.default_rng(7)
    x_matrix = rng.normal(size=(40, 20)).astype("float32")
    return AnnData(X=x_matrix, obs=_make_obs(40))


@pytest.fixture
def sparse_adata() -> AnnData:
    rng = np.random.default_rng(11)
    x_matrix = sparse.csr_matrix(rng.poisson(lam=2.0, size=(40, 20)).astype("float32"))
    return AnnData(X=x_matrix, obs=_make_obs(40))
