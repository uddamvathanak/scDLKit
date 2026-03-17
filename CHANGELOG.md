# Changelog

## Unreleased

- Added `scdlkit.foundation` with an experimental frozen scGPT embedding path for the official `whole-human` checkpoint.
- Extended `Trainer.predict_dataset(...)` to support batch-aware inference hooks for non-`x` tokenized workflows.
- Added a real-checkpoint foundation smoke script, an experimental scGPT PBMC notebook, and docs for the new embedding workflow.
- Extended the quality suite to compare the scGPT pilot against `PCA` on built-in PBMC datasets.

## 0.1.3

- Added `scdlkit.adapters` with adapter-first support for wrapping custom PyTorch reconstruction and classification modules through `Trainer`.
- Exported low-level report helpers from `scdlkit.evaluation` so custom-model workflows can save reports without internal imports.
- Added a public custom-model tutorial and docs guide showing how to wrap a raw autoencoder, evaluate it, and continue with Scanpy.

## 0.1.2

- Hardened the internal benchmark suite into a release-candidate gate around built-in Scanpy datasets.
- Added tutorial-suite execution and artifact validation so notebook outputs are checked, not only executed.
- Added missing-run detection, runtime budgets, and release-RC readiness summaries for quality evaluation.
- Added stable output-path contracts to the public notebooks and tightened docs around benchmark evidence.
- Kept the package gene-expression-focused with no public runtime API changes.

## 0.1.1

- Migrated the documentation site from MkDocs to a Sphinx-based scientific layout.
- Added a Scanpy-first tutorial path built around PBMC notebooks rendered directly in the docs site.
- Added the `tutorials` extra for notebook and Scanpy-driven walkthroughs.
- Documented a shared CPU/GPU tutorial path through `device="auto"`.
- Kept the synthetic script and notebook as lightweight smoke-test examples rather than the primary onboarding path.

## 0.1.0

- First public release of `scdlkit`.
- AnnData-native workflow for preprocessing, training, evaluation, reporting, and visualization.
- Included baseline models:
  - `autoencoder`
  - `vae`
  - `denoising_autoencoder`
  - `transformer_ae`
  - `mlp_classifier`
- Included tasks:
  - `representation`
  - `reconstruction`
  - `classification`
- Public install paths:
  - `python -m pip install scdlkit`
  - `python -m pip install scdlkit[scanpy]`
  - `python -m pip install scdlkit[notebook]`
- Release verification targets:
  - Linux
  - macOS
  - Windows
