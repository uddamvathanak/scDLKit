# scDLKit

[![CI](https://img.shields.io/github/actions/workflow/status/uddamvathanak/scDLKit/ci.yml?label=ci)](https://github.com/uddamvathanak/scDLKit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/uddamvathanak/scDLKit/docs.yml?label=docs)](https://github.com/uddamvathanak/scDLKit/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/scdlkit?cacheSeconds=300)](https://pypi.org/project/scdlkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/scdlkit?cacheSeconds=300)](https://pypi.org/project/scdlkit/)
[![License](https://img.shields.io/pypi/l/scdlkit?cacheSeconds=300)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/uddamvathanak/scDLKit?style=social)](https://github.com/uddamvathanak/scDLKit/stargazers)

Train, evaluate, compare, and visualize baseline deep-learning models for single-cell data without writing PyTorch from scratch.

## Start Here

- Documentation site: https://uddamvathanak.github.io/scDLKit/
- Primary notebook tutorial: `examples/train_vae_pbmc.ipynb`
- Install path for tutorials: `python -m pip install "scdlkit[tutorials]"`
- CPU and GPU use the same notebook path through `device="auto"`
- Secondary notebooks: `examples/compare_models_pbmc.ipynb`, `examples/classification_demo.ipynb`
- Synthetic smoke examples: `examples/first_run_synthetic.ipynb`, `examples/first_run_synthetic.py`

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

Primary tutorial install path:

```bash
python -m pip install "scdlkit[tutorials]"
```

Windows note: if you install into a deeply nested virtual environment path, Jupyter dependencies can hit Windows path-length limits. Use a short environment path such as `C:\venvs\scdlkit`, or enable Windows Long Paths if needed.

Optional extras:

```bash
python -m pip install "scdlkit[scanpy]"
python -m pip install "scdlkit[notebook]"
python -m pip install scdlkit
python -m pip install "scdlkit[dev,docs]"
```

For GPU users, install the matching PyTorch build first using the official selector:

- https://docs.pytorch.org/get-started/locally/

Then install `scdlkit[tutorials]`. The same notebook examples run on CPU or GPU with `device="auto"`.

## Scanpy Quickstart

Primary tutorial example:

```python
import scanpy as sc
from scdlkit import TaskRunner

adata = sc.datasets.pbmc3k_processed()

runner = TaskRunner(
    model="vae",
    task="representation",
    label_key="louvain",
    device="auto",
    epochs=10,
    batch_size=128,
)

runner.fit(adata)
adata.obsm["X_scdlkit_vae"] = runner.encode(adata)
```

Then continue with Scanpy:

```python
import scanpy as sc

sc.pp.neighbors(adata, use_rep="X_scdlkit_vae")
sc.tl.umap(adata)
sc.pl.umap(adata, color="louvain")
```

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
- works on CPU or GPU through `device="auto"`

Additional Scanpy-first notebooks:

- `examples/compare_models_pbmc.ipynb`: compare `autoencoder`, `vae`, and `transformer_ae`
- `examples/classification_demo.ipynb`: run the `mlp_classifier` baseline and inspect a confusion matrix

The synthetic notebook and script are still available, but they are now the smoke-test path rather than the primary researcher onboarding flow:

```bash
python -m pip install "scdlkit[notebook]"
jupyter notebook examples/first_run_synthetic.ipynb

python examples/first_run_synthetic.py
```

These write small reproducible artifacts to `artifacts/first_run_notebook/` and `artifacts/first_run/`.

## Optional contributor Conda environment

Conda is kept for contributors and demos. It is not the primary public install path for `v0.1.1`.

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
