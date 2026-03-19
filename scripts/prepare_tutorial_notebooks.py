"""Sync and optionally execute tutorial notebooks for the docs site."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
TUTORIAL_DIR = DOCS_DIR / "_tutorials"
STATIC_DIR = DOCS_DIR / "_static"

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
    "synthetic_smoke.ipynb": "classic",
}

ASSET_MAP = {
    "first_run_loss_curve.png": ROOT / "artifacts" / "first_run_notebook" / "loss_curve.png",
    "first_run_latent_pca.png": ROOT / "artifacts" / "first_run_notebook" / "latent_pca.png",
    "pbmc_vae_latent_umap.png": ROOT / "artifacts" / "pbmc_vae_quickstart" / "latent_umap.png",
    "pbmc_benchmark_comparison.png": (
        ROOT / "artifacts" / "pbmc_compare" / "benchmark_comparison.png"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        choices=("none", "published"),
        default="none",
        help="Execute synced notebooks after copying them.",
    )
    parser.add_argument(
        "--group",
        choices=("all", "classic", "foundation"),
        default="all",
        help="Subset of tutorial notebooks to sync.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional target notebook filenames to sync explicitly.",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Skip syncing static docs assets.",
    )
    return parser.parse_args()


def _selected_notebooks(group: str, only: list[str] | None) -> list[tuple[str, Path]]:
    notebooks = list(NOTEBOOK_MAP.items())
    if only:
        allowed = set(only)
        selected = [
            (target_name, source_path)
            for target_name, source_path in notebooks
            if target_name in allowed
        ]
        missing = sorted(allowed - {target_name for target_name, _ in selected})
        if missing:
            msg = (
                "Unknown tutorial notebook targets passed to --only: "
                f"{', '.join(missing)}."
            )
            raise ValueError(msg)
        return selected
    if group == "all":
        return notebooks
    return [
        (target_name, source_path)
        for target_name, source_path in notebooks
        if NOTEBOOK_GROUPS[target_name] == group
    ]


def sync_notebooks(*, execute: bool, group: str, only: list[str] | None) -> None:
    TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    for target_name, source_path in _selected_notebooks(group, only):
        if not source_path.exists():
            msg = f"Required tutorial notebook is missing: {source_path}"
            raise FileNotFoundError(msg)
        print(
            f"[prepare_tutorial_notebooks] syncing {source_path.name} -> {target_name}",
            flush=True,
        )
        notebook = nbformat.read(source_path, as_version=4)
        target_path = TUTORIAL_DIR / target_name
        if execute:
            print(
                f"[prepare_tutorial_notebooks] executing {target_name} for published docs",
                flush=True,
            )
            executor = ExecutePreprocessor(timeout=None, kernel_name="python3")
            executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
            print(
                f"[prepare_tutorial_notebooks] finished executing {target_name}",
                flush=True,
            )
        nbformat.write(notebook, target_path)
        print(f"[prepare_tutorial_notebooks] wrote {target_path}", flush=True)


def sync_assets() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    for target_name, source_path in ASSET_MAP.items():
        if not source_path.exists():
            msg = f"Required docs asset is missing: {source_path}"
            raise FileNotFoundError(msg)
        shutil.copy2(source_path, STATIC_DIR / target_name)


def main() -> None:
    args = parse_args()
    sync_notebooks(
        execute=args.execute == "published",
        group=args.group,
        only=args.only,
    )
    if not args.skip_assets:
        sync_assets()


if __name__ == "__main__":
    main()
