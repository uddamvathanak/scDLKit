# Models

scDLKit v0.1 ships with:

- `autoencoder`
- `vae`
- `denoising_autoencoder`
- `transformer_ae`
- `mlp_classifier`

All encoder-decoder models support both `representation` and `reconstruction` workflows. The classifier baseline supports `classification`.

The transformer baseline uses a patch-based tabular encoder so it remains tractable on high-dimensional expression matrices.
