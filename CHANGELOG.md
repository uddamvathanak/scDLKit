# Changelog

## Unreleased

- Added an internal quality suite for release evaluation on Scanpy built-in datasets.
- Added quickstart versus full tutorial profiles for the public notebook workflow.
- Expanded PBMC comparison guidance to include `PCA` as a classical baseline reference.
- Added maintainer-only quality gates and roadmap notes to `AGENTS.md`.

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
