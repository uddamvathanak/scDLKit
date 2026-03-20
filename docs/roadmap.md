# Roadmap

## Current phase

scDLKit is now in a cautious experimental adaptation phase with a stronger wrapper-first story.

The tutorial and API-reference hardening milestone is complete enough to support the current public workflow, including the narrow experimental scGPT bridge. The immediate priority is to make that bridge easier to use on labeled user datasets without broadening the toolkit into a general framework too early.

Current goals:

- keep the quickstart-first landing pages and tutorial learning path stable
- preserve the hardened downstream Scanpy and reconstruction tutorials
- keep the frozen scGPT bridge and annotation fine-tuning path experimental and inspectable
- add easier wrapper-first adaptation workflows without weakening benchmark or tutorial quality gates
- keep the project gene-expression-focused while the foundation path matures

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
- experimental frozen and annotation-tuned scGPT workflows as a narrow foundation-model bridge
- an easy wrapper-first scGPT annotation path for users who want a lower-code compare-predict-save loop

## Completed recently

- quickstart-first GitHub and docs landing pages
- a stronger notebook learning path instead of a flat tutorial list
- a downstream Scanpy interpretation tutorial after the model step
- a dedicated reconstruction sanity-check tutorial for predicted expression
- a narrative-first API reference
- an experimental frozen scGPT embedding bridge
- an experimental annotation-only scGPT fine-tuning path through `Trainer`
- an easy dataset-specific scGPT annotation wrapper that compares strategies and saves the best fitted runner

## Next feature phase

Target after this release: cautious continuation of experimental adaptation workflows

Planned direction:

- keep adapter-based support for user-supplied PyTorch modules through `Trainer`
- keep the built-in registry path for bundled baselines unchanged
- keep scGPT fine-tuning experimental, narrow, and easy to inspect through tutorials, reports, and wrapper artifacts
- evaluate broader adaptation strategies only after they are benchmarked against the current baselines and frozen foundation path
- defer broad foundation abstractions until the scGPT adaptation path is stable

## Deferred work

- spatial baseline support
- multimodal workflows
- broad framework-style expansion
- broad foundation-model abstraction before the scGPT adaptation path is stable
- CLI or agent workflows before the core tutorials and API docs are mature
- scverse ecosystem submission before the public workflow is easier to evaluate

Documentation quality remains a release gate, but it is no longer the only active phase. The main priority is still a trustworthy, teachable baseline toolkit rather than a broad single-cell framework.
