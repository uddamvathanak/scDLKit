# Install and Quickstart

## Public install path

Install the package from PyPI:

```bash
python -m pip install scdlkit
```

Install notebook support for the beginner tutorial notebook:

```bash
python -m pip install "scdlkit[notebook]"
```

Install Scanpy support for the PBMC tutorials:

```bash
python -m pip install "scdlkit[scanpy]"
```

## Fastest way to learn the package

Most users should start with the notebook-first path:

```bash
jupyter notebook examples/first_run_synthetic.ipynb
```

That walkthrough creates a synthetic `AnnData`, fits a baseline model with `TaskRunner`, evaluates it, and writes plots plus reports to `artifacts/first_run_notebook/`.

If you prefer a script-based first run:

```bash
python examples/first_run_synthetic.py
```

## Minimal Python API example

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

## What to open next

- For notebook workflows, go to [Tutorials](tutorials.md)
- For lower-level training control, go to [Training](training.md)
- For model selection, go to [Models](models.md)

## Optional contributor Conda setup

The repository also ships a Conda environment for contributors and demos:

```bash
conda env create -f environment.yml
conda activate scdlkit
```
