# Roadmap

## Current phase

scDLKit is in a tutorial and API reference hardening phase.

The immediate priority is not to add more modeling scope. It is to make the existing toolkit easier to judge, learn, and validate through the public documentation itself.

Current goals:

- make the GitHub landing page and docs homepage quickstart-first
- turn the notebooks into a real learning path rather than a flat list
- add a downstream Scanpy interpretation tutorial after the model step
- add a dedicated reconstruction sanity-check tutorial for predicted expression
- make the API reference narrative-first instead of autodoc-only
- keep the project gene-expression-only while the public workflow is being hardened

## What scDLKit currently emphasizes

The current public story is:

- start from processed `AnnData`
- train or run a model with scDLKit
- recover embeddings and, when supported, reconstructed expression
- continue downstream analysis in Scanpy
- use tutorials and reports to decide whether a model is behaving sensibly

This means the public workflow is still centered on:

- `TaskRunner` for the main beginner path
- `Trainer` plus adapters for lower-level extension
- experimental frozen scGPT embeddings as a narrow early foundation-model bridge

## Next feature phase

Target after this documentation milestone: cautious continuation of the experimental foundation-model path

Planned direction:

- keep adapter-based support for user-supplied PyTorch modules through `Trainer`
- keep the built-in registry path for bundled baselines unchanged
- keep frozen scGPT embeddings experimental and embedding-only
- only revisit broader fine-tuning or adaptation once the tutorials and API reference are strong enough to make those features inspectable

## Deferred work

- spatial baseline support
- multimodal workflows
- broad framework-style expansion
- broad foundation-model abstraction before the frozen scGPT path is stable
- CLI or agent workflows before the core tutorials and API docs are mature
- scverse ecosystem submission before the public workflow is easier to evaluate

The main priority remains a trustworthy, teachable baseline toolkit rather than a broad single-cell framework.
