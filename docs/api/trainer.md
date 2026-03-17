# Trainer

Use `Trainer` when you need more control than `TaskRunner` provides.

This is the right surface for:

- custom model adapters
- low-level dataset preparation
- explicit training and inference control
- batch-aware prediction flows such as experimental scGPT embeddings

Minimal example:

```python
from scdlkit import Trainer, create_model, prepare_data

prepared = prepare_data(adata, label_key="louvain")
model = create_model("vae", input_dim=prepared.input_dim, latent_dim=32)
trainer = Trainer(model=model, task="representation", device="auto", epochs=20)
trainer.fit(prepared.train, prepared.val)
predictions = trainer.predict_dataset(prepared.test or prepared.val or prepared.train)
```

```{eval-rst}
.. autoclass:: scdlkit.training.trainer.Trainer
   :members:
```
