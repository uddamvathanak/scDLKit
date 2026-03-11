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

```bash
pip install scdlkit
```

Optional extras:

```bash
pip install scdlkit[scanpy]
pip install scdlkit[dev,docs]
```

## Quickstart

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

## Examples

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
