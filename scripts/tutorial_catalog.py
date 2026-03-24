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
    "scgpt_human_pancreas_annotation.ipynb": (
        ROOT / "examples" / "scgpt_human_pancreas_annotation.ipynb"
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
    "scgpt_human_pancreas_annotation.ipynb": "foundation",
    "synthetic_smoke.ipynb": "classic",
}

NOTEBOOK_TITLES = {
    "scanpy_pbmc_quickstart.ipynb": "Scanpy PBMC quickstart",
    "downstream_scanpy_after_scdlkit.ipynb": "Downstream Scanpy after scDLKit",
    "pbmc_model_comparison.ipynb": "PBMC model comparison",
    "reconstruction_sanity_pbmc.ipynb": "Reconstruction sanity check",
    "pbmc_classification.ipynb": "PBMC classification",
    "custom_model_extension.ipynb": "Custom model extension",
    "scgpt_pbmc_embeddings.ipynb": "Experimental scGPT PBMC embeddings",
    "scgpt_cell_type_annotation.ipynb": "Experimental scGPT cell-type annotation",
    "scgpt_dataset_specific_annotation.ipynb": (
        "Experimental scGPT dataset-specific annotation"
    ),
    "scgpt_human_pancreas_annotation.ipynb": "Human-pancreas annotation quickstart",
    "synthetic_smoke.ipynb": "Synthetic smoke tutorial",
}

NOTEBOOK_SECTIONS = {
    "scanpy_pbmc_quickstart.ipynb": "Supporting workflows",
    "downstream_scanpy_after_scdlkit.ipynb": "Supporting workflows",
    "pbmc_model_comparison.ipynb": "Supporting workflows",
    "reconstruction_sanity_pbmc.ipynb": "Advanced / appendix workflows",
    "pbmc_classification.ipynb": "Advanced / appendix workflows",
    "custom_model_extension.ipynb": "Advanced / appendix workflows",
    "scgpt_pbmc_embeddings.ipynb": "Advanced / appendix workflows",
    "scgpt_cell_type_annotation.ipynb": "Experimental detail appendix",
    "scgpt_dataset_specific_annotation.ipynb": "Experimental detail appendix",
    "scgpt_human_pancreas_annotation.ipynb": "Main research task: annotation",
    "synthetic_smoke.ipynb": "Maintainer / smoke only",
}

EXECUTED_STEMS = {
    "scanpy_pbmc_quickstart.ipynb": "scanpy_pbmc_quickstart",
    "downstream_scanpy_after_scdlkit.ipynb": "downstream_scanpy_after_scdlkit",
    "pbmc_model_comparison.ipynb": "pbmc_model_comparison",
    "reconstruction_sanity_pbmc.ipynb": "reconstruction_sanity_pbmc",
    "pbmc_classification.ipynb": "pbmc_classification",
    "custom_model_extension.ipynb": "custom_model_extension",
    "scgpt_pbmc_embeddings.ipynb": "scgpt_pbmc_embeddings",
    "scgpt_cell_type_annotation.ipynb": "scgpt_cell_type_annotation",
    "scgpt_dataset_specific_annotation.ipynb": "scgpt_dataset_specific_annotation",
    "scgpt_human_pancreas_annotation.ipynb": "scgpt_human_pancreas_annotation",
    "synthetic_smoke.ipynb": "first_run_synthetic",
}

REQUIRED_ARTIFACTS = {
    "scanpy_pbmc_quickstart.ipynb": (
        ROOT / "artifacts" / "pbmc_vae_quickstart" / "report.md",
        ROOT / "artifacts" / "pbmc_vae_quickstart" / "report.csv",
        ROOT / "artifacts" / "pbmc_vae_quickstart" / "loss_curve.png",
        ROOT / "artifacts" / "pbmc_vae_quickstart" / "latent_umap.png",
    ),
    "downstream_scanpy_after_scdlkit.ipynb": (
        ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "report.md",
        ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "report.csv",
        ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "latent_umap.png",
        ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "leiden_umap.png",
        ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "marker_dotplot.png",
        ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "rank_genes_groups.csv",
    ),
    "pbmc_model_comparison.ipynb": (
        ROOT / "artifacts" / "pbmc_compare" / "benchmark_metrics.csv",
        ROOT / "artifacts" / "pbmc_compare" / "benchmark_comparison.png",
        ROOT / "artifacts" / "pbmc_compare" / "pca_reference_umap.png",
        ROOT / "artifacts" / "pbmc_compare" / "best_baseline_umap.png",
    ),
    "reconstruction_sanity_pbmc.ipynb": (
        ROOT / "artifacts" / "reconstruction_sanity_pbmc" / "report.md",
        ROOT / "artifacts" / "reconstruction_sanity_pbmc" / "report.csv",
        ROOT / "artifacts" / "reconstruction_sanity_pbmc" / "loss_curve.png",
        ROOT / "artifacts" / "reconstruction_sanity_pbmc" / "reconstruction_scatter.png",
        ROOT
        / "artifacts"
        / "reconstruction_sanity_pbmc"
        / "gene_panel_reconstruction.png",
    ),
    "pbmc_classification.ipynb": (
        ROOT / "artifacts" / "pbmc_classification" / "report.md",
        ROOT / "artifacts" / "pbmc_classification" / "report.csv",
        ROOT / "artifacts" / "pbmc_classification" / "loss_curve.png",
        ROOT / "artifacts" / "pbmc_classification" / "confusion_matrix.png",
    ),
    "custom_model_extension.ipynb": (
        ROOT / "artifacts" / "custom_model_extension" / "report.md",
        ROOT / "artifacts" / "custom_model_extension" / "report.csv",
        ROOT / "artifacts" / "custom_model_extension" / "loss_curve.png",
        ROOT / "artifacts" / "custom_model_extension" / "latent_umap.png",
    ),
    "scgpt_pbmc_embeddings.ipynb": (
        ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "report.md",
        ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "report.csv",
        ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "latent_umap.png",
        ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "linear_probe_confusion_matrix.png",
        ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "embedding_summary.json",
    ),
    "scgpt_cell_type_annotation.ipynb": (
        ROOT / "artifacts" / "scgpt_cell_type_annotation" / "report.md",
        ROOT / "artifacts" / "scgpt_cell_type_annotation" / "report.csv",
        ROOT / "artifacts" / "scgpt_cell_type_annotation" / "strategy_metrics.csv",
        ROOT / "artifacts" / "scgpt_cell_type_annotation" / "frozen_embedding_umap.png",
        ROOT / "artifacts" / "scgpt_cell_type_annotation" / "lora_embedding_umap.png",
        ROOT
        / "artifacts"
        / "scgpt_cell_type_annotation"
        / "best_strategy_confusion_matrix.png",
    ),
    "scgpt_dataset_specific_annotation.ipynb": (
        ROOT / "artifacts" / "scgpt_dataset_specific_annotation" / "report.md",
        ROOT / "artifacts" / "scgpt_dataset_specific_annotation" / "report.csv",
        ROOT / "artifacts" / "scgpt_dataset_specific_annotation" / "strategy_metrics.csv",
        ROOT
        / "artifacts"
        / "scgpt_dataset_specific_annotation"
        / "best_strategy_confusion_matrix.png",
        ROOT / "artifacts" / "scgpt_dataset_specific_annotation" / "frozen_embedding_umap.png",
        ROOT
        / "artifacts"
        / "scgpt_dataset_specific_annotation"
        / "best_strategy_embedding_umap.png",
        ROOT
        / "artifacts"
        / "scgpt_dataset_specific_annotation"
        / "best_model"
        / "manifest.json",
        ROOT
        / "artifacts"
        / "scgpt_dataset_specific_annotation"
        / "best_model"
        / "model_state.pt",
    ),
    "scgpt_human_pancreas_annotation.ipynb": (
        ROOT / "artifacts" / "scgpt_human_pancreas_annotation" / "report.md",
        ROOT / "artifacts" / "scgpt_human_pancreas_annotation" / "report.csv",
        ROOT / "artifacts" / "scgpt_human_pancreas_annotation" / "strategy_metrics.csv",
        ROOT
        / "artifacts"
        / "scgpt_human_pancreas_annotation"
        / "best_strategy_confusion_matrix.png",
        ROOT / "artifacts" / "scgpt_human_pancreas_annotation" / "frozen_embedding_umap.png",
        ROOT
        / "artifacts"
        / "scgpt_human_pancreas_annotation"
        / "best_strategy_embedding_umap.png",
        ROOT
        / "artifacts"
        / "scgpt_human_pancreas_annotation"
        / "best_model"
        / "manifest.json",
        ROOT
        / "artifacts"
        / "scgpt_human_pancreas_annotation"
        / "best_model"
        / "model_state.pt",
    ),
    "synthetic_smoke.ipynb": (
        ROOT / "artifacts" / "first_run_notebook" / "report.md",
        ROOT / "artifacts" / "first_run_notebook" / "report.csv",
        ROOT / "artifacts" / "first_run_notebook" / "loss_curve.png",
        ROOT / "artifacts" / "first_run_notebook" / "latent_pca.png",
    ),
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
