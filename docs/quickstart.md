# Quickstart

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

For a lower-level workflow:

```python
from scdlkit import Trainer, create_model, prepare_data

prepared = prepare_data(adata, label_key="cell_type")
model = create_model("vae", input_dim=prepared.input_dim, latent_dim=32)
trainer = Trainer(model=model, task="representation", epochs=15)
trainer.fit(prepared.train, prepared.val)
```
