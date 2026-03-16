# Roadmap

## Current phase

scDLKit is in an adapter-first extensibility phase on top of the earlier quality-hardening work.

Immediate goals:

- keep the public built-in API stable while adding a narrow custom-model path
- let users prototype raw PyTorch modules through `Trainer`
- preserve the Scanpy handoff workflow through `adata.obsm`
- keep the project gene-expression-only until the extension path is stable

## Current extensibility direction

Target: `v0.1.3` adapter-first custom-model support

Current direction:

- adapter-based support for user-supplied PyTorch modules through `Trainer`
- keep the built-in registry path for bundled baselines
- show custom-model integration without changing the Scanpy downstream workflow
- treat this as the bridge to later foundation-model integration rather than the foundation-model layer itself

## Later phase

Target: `v0.2.0` application and downstream analysis expansion

Planned direction:

- deeper downstream tutorials built around the latent embeddings
- stronger guidance on when to use PCA versus scDLKit baselines
- reconstruction and denoising sanity-check tutorials
- later foundation-model integration once the adapter path is stable

## Deferred work

- scverse ecosystem submission
- spatial baseline support
- multimodal workflows
- broad framework-style expansion
- foundation-model fine-tuning before the adapter path is stable

The main priority remains a trustworthy baseline toolkit rather than a broad single-cell framework.
