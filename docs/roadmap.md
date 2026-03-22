# Roadmap

## Current phase

scDLKit is now in a cautious experimental adaptation evidence phase.

The documentation and ease-of-use hardening work from `0.1.6` remains a
standing release gate, but the active product question is narrower:

- can the current wrapper-first scGPT annotation workflow show useful evidence
  beyond PBMC on another labeled human dataset?

The narrow experimental scGPT bridge remains in scope, but it still has to
satisfy the same documentation contract as the stable baseline path.

Current goals:

- keep the quickstart-first landing pages and tutorial learning path stable
- make tutorial coverage and API-reference completeness a standing release gate
- keep the top-level experimental annotation quickstart easy to discover
- keep fine-tuning and dataset adaptation visible earlier in the quickstart-facing docs for researchers with labeled data
- preserve the hardened downstream Scanpy and reconstruction tutorials
- keep the frozen scGPT bridge and annotation fine-tuning path experimental and inspectable
- add beyond-PBMC evidence without weakening benchmark or tutorial quality gates
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
- `adapt_annotation(...)` as the easiest experimental annotation path
- `Trainer` plus adapters for lower-level extension
- experimental frozen and annotation-tuned scGPT workflows as a narrow foundation-model bridge
- an easy wrapper-first scGPT annotation path for users who want a lower-code compare-predict-save loop

For researcher-facing onboarding, the public story should also answer these questions quickly:

- can I fine-tune on my labeled `AnnData` with very little code?
- can I compare frozen versus tuned strategies without writing a custom training loop?
- can I write predictions and embeddings back into `AnnData` and keep working in Scanpy?
- can I save and reload the best adapted model for later use?

## Completed recently

- quickstart-first GitHub and docs landing pages
- a stronger notebook learning path instead of a flat tutorial list
- a downstream Scanpy interpretation tutorial after the model step
- a dedicated reconstruction sanity-check tutorial for predicted expression
- a narrative-first API reference
- an experimental frozen scGPT embedding bridge
- an experimental annotation-only scGPT fine-tuning path through `Trainer`
- an easy dataset-specific scGPT annotation wrapper that compares strategies and saves the best fitted runner
- a unified top-level experimental annotation quickstart alias for easier beginner discovery

## Next feature phase

Target after this release: beyond-PBMC evidence on OpenProblems human pancreas

Planned direction:

- keep adapter-based support for user-supplied PyTorch modules through `Trainer`
- keep the built-in registry path for bundled baselines unchanged
- keep scGPT fine-tuning experimental, narrow, and easy to inspect through tutorials, reports, and wrapper artifacts
- keep the beginner annotation path low-code and CPU-practical by default
- benchmark the wrapper-first annotation path on a cached human pancreas subset in addition to PBMC
- compare `PCA + logistic regression`, frozen scGPT, head-only tuning, and LoRA without making universal superiority claims
- add a heavier manual or scheduled workflow for external annotation evidence instead of pushing that cost into normal PR CI
- evaluate broader adaptation strategies only after they are benchmarked against the current baselines and frozen foundation path
- defer broad foundation abstractions until the scGPT adaptation path is stable

## Researcher adoption loop

Near-term product and outreach work should stay close to the real researcher questions:

- study what labeled-data users actually want from fine-tuning, adaptation, and model reuse
- turn repeated user questions into quickstart examples, wrapper defaults, and short comparison tutorials
- use public before-and-after examples, small benchmark tables, and low-code snippets to explain the value of scDLKit
- share those concrete examples through repository docs, GitHub discussions, and short LinkedIn updates that point back to the quickstart and tutorial pages

## Immediate next tasks

- keep the default quickstart ladder on `frozen_probe` plus `head` unless runtime and usability data justify broadening it
- add deterministic download-and-cache support for OpenProblems human pancreas
- publish a beyond-PBMC annotation tutorial on the cached pancreas quickstart subset
- add a benchmark guide that compares PBMC and human pancreas outcomes without overclaiming
- keep turning the most common researcher questions into short docs, comparison tables, reusable examples, and release-facing evidence

## Deferred work

- spatial baseline support
- multimodal workflows
- broad framework-style expansion
- broad foundation-model abstraction before the scGPT adaptation path is stable
- CLI or agent workflows before the core tutorials and API docs are mature
- scverse ecosystem submission before the public workflow is easier to evaluate

Documentation quality remains a release gate, but it is no longer the only active phase. The main priority is still a trustworthy, teachable baseline toolkit rather than a broad single-cell framework.

## Documentation contract

Until the current surface is fully backfilled, no major new public feature should land without an explicit short-lived exception.

The standing policy is:

- every public feature needs at least one workflow tutorial
- every public feature needs an API contract page
- stable and experimental features are held to the same documentation standard
- public docs must make the right entrypoint obvious:
  - embeddings: `TaskRunner`
  - labeled annotation adaptation: `adapt_annotation(...)`
  - lower-level customization: `Trainer` plus explicit helpers
