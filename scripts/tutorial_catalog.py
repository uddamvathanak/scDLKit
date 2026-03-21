"""Lightweight tutorial metadata shared by docs and validation scripts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_MAP = {
    "scanpy_pbmc_quickstart.ipynb": ROOT / "examples" / "train_vae_pbmc.ipynb",
    "downstream_scanpy_after_scdlkit.ipynb": (
        ROOT / "examples" / "downstream_scanpy_after_scdlkit.ipynb"
    ),
    "pbmc_model_comparison.ipynb": ROOT / "examples" / "compare_models_pbmc.ipynb",
    "reconstruction_sanity_pbmc.ipynb": ROOT / "examples" / "reconstruction_sanity_pbmc.ipynb",
    "pbmc_classification.ipynb": ROOT / "examples" / "classification_demo.ipynb",
    "custom_model_extension.ipynb": ROOT / "examples" / "custom_model_extension.ipynb",
    "scgpt_pbmc_embeddings.ipynb": ROOT / "examples" / "scgpt_pbmc_embeddings.ipynb",
    "scgpt_cell_type_annotation.ipynb": ROOT / "examples" / "scgpt_cell_type_annotation.ipynb",
    "scgpt_dataset_specific_annotation.ipynb": (
        ROOT / "examples" / "scgpt_dataset_specific_annotation.ipynb"
    ),
    "synthetic_smoke.ipynb": ROOT / "examples" / "first_run_synthetic.ipynb",
}

NOTEBOOK_GROUPS = {
    "scanpy_pbmc_quickstart.ipynb": "classic",
    "downstream_scanpy_after_scdlkit.ipynb": "classic",
    "pbmc_model_comparison.ipynb": "classic",
    "reconstruction_sanity_pbmc.ipynb": "classic",
    "pbmc_classification.ipynb": "classic",
    "custom_model_extension.ipynb": "classic",
    "scgpt_pbmc_embeddings.ipynb": "foundation",
    "scgpt_cell_type_annotation.ipynb": "foundation",
    "scgpt_dataset_specific_annotation.ipynb": "foundation",
    "synthetic_smoke.ipynb": "classic",
}

ASSET_MAP = {
    "first_run_loss_curve.png": ROOT / "artifacts" / "first_run_notebook" / "loss_curve.png",
    "first_run_latent_pca.png": ROOT / "artifacts" / "first_run_notebook" / "latent_pca.png",
    "pbmc_vae_latent_umap.png": ROOT / "artifacts" / "pbmc_vae_quickstart" / "latent_umap.png",
    "pbmc_downstream_leiden_umap.png": (
        ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "leiden_umap.png"
    ),
    "pbmc_benchmark_comparison.png": (
        ROOT / "artifacts" / "pbmc_compare" / "benchmark_comparison.png"
    ),
}

TUTORIAL_IDS = tuple(path.removesuffix(".ipynb") for path in NOTEBOOK_MAP)
