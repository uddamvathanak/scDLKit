# Roadmap

## Current phase

scDLKit is in a quality-hardening phase rather than a model-zoo expansion phase.

Immediate goals:

- stabilize the public API and default behaviors
- lengthen the public tutorials without changing the overall workflow
- benchmark the toolkit itself on small built-in Scanpy datasets
- keep the project gene-expression-only until the quality gates stay green

## Immediate milestone

Target: `v0.1.2` quality release

Planned deliverables:

- quickstart and full profiles in the public notebooks
- explicit internal quality-suite scripts and benchmark summaries
- `PCA` as the classical reference baseline in the comparison tutorial
- regression checks for PBMC latent quality and classification quality

## Next phase

Target: `v0.1.3` extensibility release

Planned direction:

- adapter-based support for user-supplied PyTorch modules
- keep the built-in registry path for bundled baselines
- show custom-model integration without changing the Scanpy downstream workflow

## Later phase

Target: `v0.2.0` application and downstream analysis expansion

Planned direction:

- deeper downstream tutorials built around the latent embeddings
- stronger guidance on when to use PCA versus scDLKit baselines
- reconstruction and denoising sanity-check tutorials

## Deferred work

- scverse ecosystem submission
- spatial baseline support
- multimodal workflows
- broad framework-style expansion

The main priority remains a trustworthy baseline toolkit rather than a broad single-cell framework.
