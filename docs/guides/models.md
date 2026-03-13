# Models

scDLKit v0.1.1 remains deliberately baseline-focused.

## Available models

- `autoencoder`
- `vae`
- `denoising_autoencoder`
- `transformer_ae`
- `mlp_classifier`

## Supported tasks

- `representation`
- `reconstruction`
- `classification`

## Typical selection

- Use `vae` for the main representation-learning tutorial path.
- Use `autoencoder` for a simpler reconstruction baseline.
- Use `transformer_ae` when you want an attention-based baseline in comparisons.
- Use `mlp_classifier` for direct supervised classification from expression features.

## Example

```python
from scdlkit import TaskRunner

runner = TaskRunner(
    model="vae",
    task="representation",
    latent_dim=32,
    epochs=10,
    device="auto",
)
```
