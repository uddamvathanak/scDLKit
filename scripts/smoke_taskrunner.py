"""Cross-platform smoke test for the public scDLKit API."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from scdlkit import TaskRunner


def main() -> None:
    rng = np.random.default_rng(42)
    x_matrix = rng.normal(size=(20, 12)).astype("float32")
    obs = pd.DataFrame({"cell_type": ["a"] * 10 + ["b"] * 10})
    adata = AnnData(X=x_matrix, obs=obs)
    runner = TaskRunner(
        model="vae",
        task="representation",
        epochs=1,
        batch_size=4,
        latent_dim=4,
        hidden_dims=(8, 6),
        label_key="cell_type",
    )
    runner.fit(adata)
    metrics = runner.evaluate()
    if "pearson" not in metrics:
        raise RuntimeError("Smoke test did not produce the expected metric output.")
    print("scDLKit smoke test passed.")


if __name__ == "__main__":
    main()
