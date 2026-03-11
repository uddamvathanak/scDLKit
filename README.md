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

## Installation

## Conda Setup (Recommended)

Official installers:

- Miniconda install guide: https://www.anaconda.com/docs/getting-started/miniconda/install
- Anaconda Distribution download: https://www.anaconda.com/download

Recommended path:

1. Install Miniconda or Anaconda Distribution.
2. Open Anaconda Prompt.
3. Create the project environment from the repo root:

```bash
conda env create -f environment.yml
conda activate scdlkit
```

4. Run the first simple example:

```bash
python examples/first_run_synthetic.py
```

This writes a report, checkpoint, loss curve, and latent PCA plot to `artifacts/first_run/`.

If the environment already exists, update it with:

```bash
conda env update -n scdlkit -f environment.yml
conda activate scdlkit
```

### Optional Notebook Extras

The PBMC notebooks use `scanpy` and Jupyter. Install them only if you want the notebook demos:

```bash
python -m pip install scanpy jupyter
```

You can then run:

```bash
jupyter notebook
```

## Pip Install

```bash
pip install scdlkit
```

Optional extras:

```bash
pip install scdlkit[scanpy]
pip install scdlkit[dev,docs]
```

## Quickstart

Fastest first run from Conda:

```bash
conda activate scdlkit
python examples/first_run_synthetic.py
```

Python API:

```python
from scdlkit import TaskRunner

runner = TaskRunner(
    model="vae",
    task="representation",
    latent_dim=32,
    epochs=25,
    batch_size=256,
    label_key="cell_type",
    batch_key="batch",
)

runner.fit(adata)
metrics = runner.evaluate()
runner.plot_losses()
runner.plot_latent(method="umap", color="label")
runner.save_report("artifacts/report.md")
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

## Supported Models

- `autoencoder`
- `vae`
- `denoising_autoencoder`
- `transformer_ae`
- `mlp_classifier`

## Supported Tasks

- `representation`
- `reconstruction`
- `classification`

## Documentation

Project documentation is configured for GitHub Pages with MkDocs Material:

- Docs site: https://uddamvathanak.github.io/scDLKit/
- API reference: `docs/api.md`
- Example notebooks: `examples/`

### GitHub Pages Setup

The docs workflow expects GitHub Pages to be enabled once at the repository level.

1. Open `Settings -> Pages` for this repo:
   `https://github.com/uddamvathanak/scDLKit/settings/pages`
2. Under `Build and deployment`, set `Source` to `GitHub Actions`.
3. Save the setting.
4. Re-run the `docs` workflow.

Without that one-time setting, `actions/deploy-pages` returns a `404` when it tries to create the deployment.

## Examples

- `examples/first_run_synthetic.py`
- `examples/train_vae_pbmc.ipynb`
- `examples/compare_models_pbmc.ipynb`
- `examples/classification_demo.ipynb`

## Roadmap

`v0.1`

- Expanded core workflow with training, evaluation, reporting, and plotting.
- PyPI packaging and CI.
- Docs site and reproducible notebooks.

`v0.2`

- CLI and YAML config support.
- Graph-based models and richer benchmarking helpers.
- More task-specific extensions.

## Citation

If you use `scDLKit`, cite the software entry in [`CITATION.cff`](CITATION.cff).
