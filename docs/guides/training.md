# Training

`TaskRunner` is the beginner-facing training interface. `Trainer` is the lower-level entry point when you need more control.

For `v0.1.3`, `Trainer` is also the supported surface for custom wrapped PyTorch models through `scdlkit.adapters`.

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
    epochs=20,
    batch_size=128,
)
```

If CUDA is available, training will use it. If not, the same notebook continues on CPU.

For single-cell VAE tutorials, it is often worth reducing `kl_weight` from the
vanilla setting so the latent space does not over-regularize.

The public notebooks now use a two-profile strategy:

- `quickstart`: shorter CPU-friendly runs that are still expected to produce sensible latent structure
- `full`: longer runs with the same code path when you want stronger qualitative separation before interpreting the latent space

## Lower-level training

```python
from scdlkit import Trainer, create_model, prepare_data

prepared = prepare_data(adata, label_key="louvain")
model = create_model("vae", input_dim=prepared.input_dim, latent_dim=32)
trainer = Trainer(model=model, task="representation", device="auto", epochs=20)
trainer.fit(prepared.train, prepared.val)
```

## Custom PyTorch modules

If you want to prototype your own `nn.Module`, wrap it through `scdlkit.adapters` and keep using the same `Trainer` workflow.

```python
from scdlkit import Trainer, prepare_data
from scdlkit.adapters import wrap_reconstruction_module

prepared = prepare_data(adata, label_key="louvain")
wrapped = wrap_reconstruction_module(
    module=my_autoencoder,
    input_dim=prepared.input_dim,
    supported_tasks=("representation", "reconstruction"),
)

trainer = Trainer(model=wrapped, task="representation", device="auto", epochs=10)
trainer.fit(prepared.train, prepared.val)
```

This first adapter release supports modules that ultimately consume only `x` from the batch. Full-batch custom forward plumbing and foundation-model integrations remain future work.
