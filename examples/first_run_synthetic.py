"""Minimal synthetic smoke example for a fresh scDLKit install."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from scdlkit import TaskRunner

matplotlib.use("Agg")


def make_synthetic_adata(
    n_cells: int = 120,
    n_genes: int = 32,
    seed: int = 42,
) -> AnnData:
    """Build a small synthetic AnnData object for a first run."""

    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_cells, n_genes)).astype("float32")
    labels = np.array(["T-cell"] * (n_cells // 2) + ["B-cell"] * (n_cells // 2))
    batches = np.array(["batch_a"] * (n_cells // 4) + ["batch_b"] * (n_cells // 4))[
        : n_cells // 2
    ]
    batches = np.concatenate([batches, batches])
    signal = np.where(labels[:, None] == "T-cell", 1.0, -1.0).astype("float32")
    x_matrix = base + 0.6 * signal
    obs = pd.DataFrame(
        {"cell_type": labels, "batch": batches},
        index=[f"cell_{idx}" for idx in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{idx}" for idx in range(n_genes)])
    return AnnData(X=x_matrix, obs=obs, var=var)


def main() -> None:
    output_dir = Path("artifacts/first_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = make_synthetic_adata()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")

    runner = TaskRunner(
        model="autoencoder",
        task="representation",
        latent_dim=8,
        hidden_dims=(64, 32),
        epochs=5,
        batch_size=16,
        lr=1e-3,
        label_key="cell_type",
        batch_key="batch",
        device="auto",
        output_dir=str(output_dir),
    )
    runner.fit(adata)
    metrics = runner.evaluate()

    loss_fig, _ = runner.plot_losses()
    loss_fig.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")

    latent_fig, _ = runner.plot_latent(method="pca", color="cell_type")
    latent_fig.savefig(output_dir / "latent_pca.png", dpi=150, bbox_inches="tight")

    runner.save_report(output_dir / "report.md")

    print("scDLKit first run completed.")
    print(f"Artifacts written to: {output_dir.resolve()}")
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
