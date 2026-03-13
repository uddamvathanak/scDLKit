# Quickstart

## Public install path

Install the package from PyPI:

```bash
python -m pip install scdlkit
```

Install the notebook extra if you want to execute the beginner notebook:

```bash
python -m pip install "scdlkit[notebook]"
```

Install the Scanpy extra if you want the PBMC notebooks:

```bash
python -m pip install "scdlkit[scanpy]"
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
```

## Repo examples

If you cloned the repository, the smallest end-to-end example is:

```bash
python examples/first_run_synthetic.py
```

If you want the beginner notebook after cloning the repo:

```bash
jupyter notebook examples/first_run_synthetic.ipynb
```

## Optional contributor Conda setup

The repository also ships a Conda environment for contributors and demos:

```bash
conda env create -f environment.yml
conda activate scdlkit
```
