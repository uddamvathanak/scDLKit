# Quickstart

## Recommended Conda setup

```bash
conda env create -f environment.yml
conda activate scdlkit
python examples/first_run_synthetic.py
```

This is the smallest end-to-end example and writes artifacts to `artifacts/first_run/`.

If you prefer a notebook, use:

```bash
jupyter notebook examples/first_run_synthetic.ipynb
```

The default Conda environment includes the notebook dependencies for this walkthrough.

If you created `scdlkit` from an older environment file, recreate the environment once before using the notebook.

## Python API

```python
from scdlkit import TaskRunner

runner = TaskRunner(
    model="vae",
    task="representation",
    latent_dim=32,
    epochs=15,
    batch_size=128,
    label_key="cell_type",
)

runner.fit(adata)
metrics = runner.evaluate()
runner.plot_losses()
runner.plot_latent(method="umap", color="label")
runner.save_report("artifacts/vae_report.md")
```

## Lower-level workflow

For more control:

```python
from scdlkit import Trainer, create_model, prepare_data

prepared = prepare_data(adata, label_key="cell_type")
model = create_model("vae", input_dim=prepared.input_dim, latent_dim=32)
trainer = Trainer(model=model, task="representation", epochs=15)
trainer.fit(prepared.train, prepared.val)
```

## Optional PBMC notebook extras

The beginner notebook works from the default Conda environment. The PBMC notebooks still need Scanpy:

```bash
python -m pip install scanpy
```
