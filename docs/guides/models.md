# Models

scDLKit remains deliberately baseline-focused.

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
  The public PBMC comparison tutorial uses a compact CPU-friendly transformer
  setup so the attention baseline stays practical in docs and CI.
- Use `mlp_classifier` for direct supervised classification from expression features.
- Use `PCA` in the comparison tutorial as a classical reference point, even though it is not part of the built-in model registry.

For single-cell representation learning, a lighter VAE `kl_weight` often gives
better latent separation than a vanilla `beta=1` setup.

A good quickstart latent result should show broad PBMC groups separating into visible regions. If the latent space turns into a single dense mixed cloud, the training setup is too aggressive or too short for interpretation.

## Example

```python
from scdlkit import TaskRunner

runner = TaskRunner(
    model="vae",
    task="representation",
    latent_dim=32,
    epochs=20,
    device="auto",
)
```
