# scDLKit

[![CI](https://img.shields.io/github/actions/workflow/status/uddamvathanak/scDLKit/ci.yml?label=ci)](https://github.com/uddamvathanak/scDLKit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/uddamvathanak/scDLKit/docs.yml?label=docs)](https://github.com/uddamvathanak/scDLKit/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/scdlkit?cacheSeconds=300)](https://pypi.org/project/scdlkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/scdlkit?cacheSeconds=300)](https://pypi.org/project/scdlkit/)
[![License](https://img.shields.io/pypi/l/scdlkit?cacheSeconds=300)](LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/scdlkit)](https://pypistats.org/packages/scdlkit)
[![GitHub stars](https://img.shields.io/github/stars/uddamvathanak/scDLKit?style=social)](https://github.com/uddamvathanak/scDLKit/stargazers)

Train, evaluate, compare, and visualize baseline deep-learning models for single-cell data without writing PyTorch from scratch.

## Quick Start

Start here if you want the shortest path from `AnnData` to a learned embedding and, for reconstruction-capable models, predicted or reconstructed gene-expression values:

1. load an `AnnData`
2. fit a baseline model
3. get the learned embedding
4. optionally get predicted or reconstructed gene-expression values
5. continue in Scanpy

```python
import scanpy as sc
from scdlkit import TaskRunner

adata = sc.datasets.pbmc3k_processed()

runner = TaskRunner(
    model="vae",
    task="representation",
    label_key="louvain",
    device="auto",
    epochs=20,
    batch_size=128,
    model_kwargs={"kl_weight": 1e-3},
)

runner.fit(adata)

# Cell embedding for downstream Scanpy analysis.
adata.obsm["X_scdlkit_vae"] = runner.encode(adata)

# Predicted / reconstructed gene expression for reconstruction-capable models.
predicted_expression = runner.reconstruct(adata)
```

Then keep the normal Scanpy path:

```python
sc.pp.neighbors(adata, use_rep="X_scdlkit_vae")
sc.tl.umap(adata)
sc.pl.umap(adata, color="louvain")
```

Notes:

- `runner.encode(...)` returns the latent embedding.
- `runner.reconstruct(...)` returns reconstructed gene-expression values for reconstruction-capable models such as `autoencoder`, `vae`, `denoising_autoencoder`, and `transformer_ae`.
- `runner.predict(...)` remains backward compatible, but `reconstruct(...)` is the clearer public path for reconstructed expression.
- Classification models return class predictions instead of reconstructed expression.
- Frozen scGPT in the experimental foundation path exposes embeddings only, not reconstructed expression.

What you get from this quickstart:

- a learned embedding in `adata.obsm`
- reconstructed gene-expression values when the model supports them
- training metrics and saved reports
- a direct continuation path into Scanpy

## Start Here

- Documentation site: https://uddamvathanak.github.io/scDLKit/
- Primary notebook tutorial: `examples/train_vae_pbmc.ipynb`
- Install path for tutorials: `python -m pip install "scdlkit[tutorials]"`
- Experimental foundation path: `python -m pip install "scdlkit[foundation,tutorials]"`
- CPU and GPU use the same notebook path through `device="auto"`
- Core learning path: quickstart -> downstream Scanpy -> comparison -> reconstruction sanity check
- Secondary notebooks: `examples/compare_models_pbmc.ipynb`, `examples/classification_demo.ipynb`
- Downstream Scanpy notebook: `examples/downstream_scanpy_after_scdlkit.ipynb`
- Reconstruction notebook: `examples/reconstruction_sanity_pbmc.ipynb`
- Custom model notebook: `examples/custom_model_extension.ipynb`
- Experimental foundation notebook: `examples/scgpt_pbmc_embeddings.ipynb`
- Experimental annotation fine-tuning notebook: `examples/scgpt_cell_type_annotation.ipynb`
- Synthetic smoke examples: `examples/first_run_synthetic.ipynb`, `examples/first_run_synthetic.py`

## Why scDLKit

- AnnData-native workflow for single-cell users.
- Baseline-first model zoo: AE, VAE, DAE, Transformer AE, and MLP classification.
- Built-in training, evaluation, comparison, and plotting.
- Reproducible reports and notebooks for portfolio-ready demonstrations.
- Built-in benchmark gates on small Scanpy datasets before tutorial defaults change.
- Gene-expression-focused scope while the core toolkit stabilizes.
- Experimental frozen scGPT embeddings for human PBMC workflows.
- Experimental scGPT annotation fine-tuning with head-only and LoRA strategies.

## Supported platforms

- Linux: supported
- macOS: supported
- Windows: supported

## Installation

Primary tutorial install path:

```bash
python -m pip install "scdlkit[tutorials]"
```

Windows note: if you install into a deeply nested virtual environment path, Jupyter dependencies can hit Windows path-length limits. Use a short environment path such as `C:\venvs\scdlkit`, or enable Windows Long Paths if needed.

Optional extras:

```bash
python -m pip install "scdlkit[scanpy]"
python -m pip install "scdlkit[notebook]"
python -m pip install "scdlkit[foundation]"
python -m pip install scdlkit
python -m pip install "scdlkit[dev,docs]"
```

For GPU users, install the matching PyTorch build first using the official selector:

- https://docs.pytorch.org/get-started/locally/

Then install `scdlkit[tutorials]`. The same notebook examples run on CPU or GPU with `device="auto"`.

## Scanpy Quickstart

Primary tutorial example. The notebook uses a quickstart profile by default and exposes a full profile in its first config cell:

- `quickstart`: CPU-friendly, docs-friendly, reproducible
- `full`: longer run for stronger qualitative separation

For the PBMC quickstart, use a light VAE KL term so the latent UMAP preserves broad
cell-type structure instead of collapsing into a uniform blob. A healthy result should
show broad cell-type groups as visibly separated regions rather than a single mixed cloud.

## Notebook-First Examples

Most researchers should start with the Scanpy PBMC quickstart:

```bash
python -m pip install "scdlkit[tutorials]"
jupyter notebook examples/train_vae_pbmc.ipynb
```

This notebook:

- loads PBMC data through Scanpy
- trains a VAE baseline with scDLKit
- writes the latent representation into `adata.obsm`
- continues with Scanpy neighbors and UMAP
- points to the downstream Scanpy and reconstruction tutorials for the next interpretation steps
- explains the quickstart versus full tutorial profiles
- works on CPU or GPU through `device="auto"`

Additional Scanpy-first notebooks:

- `examples/downstream_scanpy_after_scdlkit.ipynb`: take the scDLKit embedding through Leiden clustering, marker ranking, dotplots, and coarse annotation
- `examples/compare_models_pbmc.ipynb`: compare `PCA`, `autoencoder`, `vae`, and `transformer_ae`
- `examples/reconstruction_sanity_pbmc.ipynb`: inspect reconstructed gene-expression outputs with a dedicated reconstruction baseline
- `examples/classification_demo.ipynb`: run the `mlp_classifier` baseline and inspect a confusion matrix
- `examples/custom_model_extension.ipynb`: wrap a raw PyTorch autoencoder and train it through `Trainer`
- `examples/scgpt_pbmc_embeddings.ipynb`: run the experimental frozen `whole-human` scGPT embedding workflow and return to Scanpy through `adata.obsm`
- `examples/scgpt_cell_type_annotation.ipynb`: compare `PCA + logistic regression`, frozen scGPT, head-only tuning, and LoRA tuning for labeled PBMC annotation

The synthetic notebook and script are still available, but they are now the smoke-test path rather than the primary researcher onboarding flow:

```bash
python -m pip install "scdlkit[notebook]"
jupyter notebook examples/first_run_synthetic.ipynb

python examples/first_run_synthetic.py
```

These write small reproducible artifacts to `artifacts/first_run_notebook/` and `artifacts/first_run/`.

## Optional contributor Conda environment

Conda is kept for contributors and demos. It is not the primary public install path.

Official installers:

- Miniconda install guide: https://www.anaconda.com/docs/getting-started/miniconda/install
- Anaconda Distribution download: https://www.anaconda.com/download

From the repo root:

```bash
conda env create -f environment.yml
conda activate scdlkit
```

## Core APIs

High-level:

```python
from scdlkit import TaskRunner
```

Lower-level:

```python
from scdlkit import Trainer, create_model, prepare_data
```

Custom-model adapters:

```python
from scdlkit.adapters import wrap_classification_module, wrap_reconstruction_module
```

Custom wrapped models are supported through `Trainer` first. `TaskRunner` remains the built-in high-level path for bundled scDLKit models.

Experimental foundation helpers:

```python
from scdlkit.foundation import load_scgpt_model, prepare_scgpt_data
```

Experimental scGPT annotation tuning:

```python
from scdlkit.foundation import (
    load_scgpt_annotation_model,
    prepare_scgpt_data,
    split_scgpt_data,
)
```

Comparison:

```python
from scdlkit import compare_models

benchmark = compare_models(
    adata,
    models=["autoencoder", "vae", "transformer_ae"],
    task="representation",
    shared_kwargs={"epochs": 10, "label_key": "cell_type"},
    output_dir="artifacts/compare",
)
```

## Supported models

- `autoencoder`
- `vae`
- `denoising_autoencoder`
- `transformer_ae`
- `mlp_classifier`

## Supported tasks

- `representation`
- `reconstruction`
- `classification`

## Current scope

- Gene-expression baselines for AnnData workflows
- Scanpy-first tutorial and downstream embedding usage
- Built-in deep-learning baselines plus classical comparison context in notebooks
- Adapter-based custom PyTorch model support through `Trainer`
- Experimental scGPT frozen embedding support for human PBMC workflows
- Experimental scGPT annotation fine-tuning for labeled human PBMC workflows through `Trainer`

Broader foundation-model support, full-backbone fine-tuning, spatial omics, and multimodal workflows remain future work once the gene-expression toolkit quality gates stay stable.

## Documentation

Project documentation is published as a Sphinx-based scientific docs site:

- Docs site: https://uddamvathanak.github.io/scDLKit/
- Tutorials: Scanpy-first notebook walkthroughs rendered in the docs site
- API reference: `docs/api/index.md`
- Example notebooks: `examples/`

### GitHub Pages setup

The docs workflow expects GitHub Pages to be enabled once at the repository level.

1. Open `Settings -> Pages` for this repo:
   `https://github.com/uddamvathanak/scDLKit/settings/pages`
2. Under `Build and deployment`, set `Source` to `GitHub Actions`.
3. Save the setting.
4. Re-run the `docs` workflow.

Without that one-time setting, GitHub returns a `404` when `actions/configure-pages` or `actions/deploy-pages` tries to access the Pages site.

### Optional automatic Pages enablement

If you want the workflow to bootstrap Pages automatically instead of doing the one-time manual setup:

1. Create a repository secret named `PAGES_ENABLEMENT_TOKEN`.
2. Use a Personal Access Token with `repo` scope or Pages write permission.
3. Re-run the `docs` workflow.

## Release flow

- Stage to TestPyPI first with `release-testpypi.yml`.
- Publish the final release from a `v*` tag with `release.yml`.
- Use trusted publishing instead of long-lived PyPI API tokens.
- See [`RELEASING.md`](RELEASING.md) for the full checklist.

## Examples

- `examples/train_vae_pbmc.ipynb` is the primary Scanpy-first notebook tutorial.
- `examples/compare_models_pbmc.ipynb` compares `autoencoder`, `vae`, and `transformer_ae` on PBMC data.
- `examples/classification_demo.ipynb` covers the `mlp_classifier` workflow and confusion-matrix reporting.
- `examples/first_run_synthetic.ipynb` is the secondary smoke-test notebook with minimal setup.
- `examples/first_run_synthetic.py` is the secondary smoke-test script.

## Roadmap

Immediate roadmap target:

- keep the built-in `TaskRunner` story stable for bundled baselines
- preserve adapter-first custom-model support through `Trainer`
- add a narrow experimental scGPT annotation fine-tuning workflow through `Trainer`
- keep the toolkit gene-expression-focused while the foundation path hardens

Released so far:

`v0.1`

- Expanded core workflow with training, evaluation, reporting, and plotting.
- Staged TestPyPI and PyPI publishing.
- Cross-platform smoke validation and reproducible notebooks.

Later:

- broader foundation-model fine-tuning beyond annotation once the experimental scGPT path is stable
- spatial baselines only after the gene-expression toolkit is stable

## Citation

If you use `scDLKit`, cite the software entry in [`CITATION.cff`](CITATION.cff).
