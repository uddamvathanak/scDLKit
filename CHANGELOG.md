# Changelog

## 0.1.7

- Added beyond-PBMC experimental annotation evidence on cached OpenProblems human pancreas subsets while keeping the public beginner surface unchanged.
- Added a heavier external annotation evidence workflow, a pancreas tutorial, and a benchmark guide so the wrapper-first scGPT annotation path can be evaluated on a non-PBMC human dataset.
- Normalized the live OpenProblems pancreas schema into the canonical internal annotation fields expected by the cached subset builder and public tutorial path.
- Hardened the heavy pancreas evidence workflow by reusing saved trainable checkpoints instead of retraining the best model and by isolating the heavier path outside normal PR CI.
- Updated GitHub Actions maintenance dependencies to the current action major versions and raised the heavy evidence workflow timeout so the beyond-PBMC path can complete cleanly.
- Refreshed the project logo, docs logo mark, and favicon assets for the 0.1.7 fine-tuning release.

## 0.1.6

- Added a unified experimental annotation quickstart at the top-level package with `adapt_annotation(...)`, `inspect_annotation_data(...)`, and `AnnotationRunner`.
- Added a wrapper-first scGPT annotation adaptation workflow that supports inspect, compare, annotate, save, and reload behavior around labeled human `AnnData`.
- Tightened the beginner default strategy ladder to `frozen_probe` plus `head`, while keeping LoRA available by explicit opt-in.
- Hardened wrapper reporting, saved-runner behavior, and the foundation annotation smoke path for the new beginner alias route.
- Added a docs contract registry, stricter API contract pages, tutorial-to-API linking, and CI/docs enforcement for public feature completeness.
- Decoupled the docs contract checker from notebook-only dependencies through a lightweight shared tutorial catalog so lean CI jobs stay installable.

## 0.1.5

- Added `scdlkit.foundation` with an experimental frozen scGPT embedding path for the official `whole-human` checkpoint.
- Added experimental scGPT annotation fine-tuning for labeled human scRNA-seq through `Trainer`, including frozen linear-probe, head-only, and LoRA comparison workflows.
- Added tokenized scGPT data preparation and split helpers for annotation workflows.
- Extended `Trainer.predict_dataset(...)` to support batch-aware inference hooks for non-`x` tokenized workflows used by the foundation path.
- Added foundation smoke scripts, an experimental scGPT PBMC embedding tutorial, and an experimental scGPT cell-type annotation tutorial.
- Hardened the public tutorial path with downstream Scanpy and reconstruction sanity-check notebooks, quickstart-first landing pages, and a narrative-first API reference.
- Extended the quality suite and CI workflows to validate the experimental foundation path and its released tutorial story.

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
