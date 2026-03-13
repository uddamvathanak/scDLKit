# scDLKit

[![CI](https://img.shields.io/github/actions/workflow/status/uddamvathanak/scDLKit/ci.yml?label=ci)](https://github.com/uddamvathanak/scDLKit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/uddamvathanak/scDLKit/docs.yml?label=docs)](https://github.com/uddamvathanak/scDLKit/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/scdlkit)](https://pypi.org/project/scdlkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/scdlkit)](https://pypi.org/project/scdlkit/)
[![License](https://img.shields.io/pypi/l/scdlkit)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/uddamvathanak/scDLKit?style=social)](https://github.com/uddamvathanak/scDLKit/stargazers)
[![Downloads](https://static.pepy.tech/badge/scdlkit)](https://pepy.tech/projects/scdlkit)

Train, evaluate, compare, and visualize baseline deep-learning models for single-cell data without writing PyTorch from scratch.

## Why scDLKit

- AnnData-native workflow for single-cell users.
- Baseline-first model zoo: AE, VAE, DAE, Transformer AE, and MLP classification.
- Built-in training, evaluation, comparison, and plotting.
- Reproducible reports and notebooks for portfolio-ready demonstrations.
- Extensible registry-based architecture for custom models and future tasks.

## Supported platforms

- Linux: supported
- macOS: supported
- Windows: supported

## Installation

Primary public install path:

```bash
python -m pip install scdlkit
```

Optional extras:

```bash
python -m pip install "scdlkit[scanpy]"
python -m pip install "scdlkit[notebook]"
python -m pip install "scdlkit[dev,docs]"
```

## Quickstart

Smallest package-level run:

```python
import numpy as np
import pandas as pd
from anndata import AnnData
from scdlkit import TaskRunner

X = np.random.rand(120, 32).astype("float32")
obs = pd.DataFrame({"cell_type": ["T-cell"] * 60 + ["B-cell"] * 60})
adata = AnnData(X=X, obs=obs)

runner = TaskRunner(
    model="vae",
    task="representation",
    latent_dim=8,
    epochs=5,
    batch_size=16,
    label_key="cell_type",
)

runner.fit(adata)
metrics = runner.evaluate()
runner.plot_losses()
```

## Repo examples

If you cloned the repository, the easiest end-to-end demo is:

```bash
python examples/first_run_synthetic.py
```

This writes a report, checkpoint, loss curve, and latent PCA plot to `artifacts/first_run/`.

If you want the beginner notebook after cloning the repo:

```bash
python -m pip install "scdlkit[notebook]"
jupyter notebook examples/first_run_synthetic.ipynb
```

The heavier notebooks still need Scanpy:

```bash
python -m pip install "scdlkit[scanpy]"
```

## Optional contributor Conda environment

Conda is kept for contributors and demos. It is not the primary public install path for `v0.1.0`.

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

## Documentation

Project documentation is configured for GitHub Pages with MkDocs Material:

- Docs site: https://uddamvathanak.github.io/scDLKit/
- API reference: `docs/api.md`
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

- `examples/first_run_synthetic.ipynb` is the easiest notebook walkthrough.
- `examples/first_run_synthetic.py` is the easiest script walkthrough.
- `examples/train_vae_pbmc.ipynb`
- `examples/compare_models_pbmc.ipynb`
- `examples/classification_demo.ipynb`

## Roadmap

`v0.1`

- Expanded core workflow with training, evaluation, reporting, and plotting.
- Staged TestPyPI and PyPI publishing.
- Cross-platform smoke validation and reproducible notebooks.

`v0.2`

- CLI and YAML config support.
- Graph-based models and richer benchmarking helpers.
- More task-specific extensions.

## Citation

If you use `scDLKit`, cite the software entry in [`CITATION.cff`](CITATION.cff).
