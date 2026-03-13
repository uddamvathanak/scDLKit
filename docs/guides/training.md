# Training

`TaskRunner` is the beginner-facing training interface. `Trainer` is the lower-level entry point when you need more control.

## Key training features

- device auto-detection
- optional mixed precision on CUDA
- early stopping
- best-checkpoint restoration
- reproducibility through seeding

## One CPU/GPU path

Use the same code on both CPU and GPU:

```python
runner = TaskRunner(
    model="vae",
    task="representation",
    device="auto",
    epochs=10,
    batch_size=128,
)
```

If CUDA is available, training will use it. If not, the same notebook continues on CPU.

## Lower-level training

```python
from scdlkit import Trainer, create_model, prepare_data

prepared = prepare_data(adata, label_key="louvain")
model = create_model("vae", input_dim=prepared.input_dim, latent_dim=32)
trainer = Trainer(model=model, task="representation", device="auto", epochs=10)
trainer.fit(prepared.train, prepared.val)
```
