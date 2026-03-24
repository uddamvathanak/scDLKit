# Milestone 5: cross-model PEFT expansion and self-supervised path

Status: planned

## Objective

Complete Pillar 1 of the abstract: a unified adaptation interface across
multiple foundation models and PEFT strategies, including a self-supervised
adaptation path for unlabeled datasets.

## Why this matters for the paper

The abstract's most ambitious claim is a "single training interface across
multiple foundation models and multiple PEFT strategies." That claim also
includes a self-supervised path — adapting to unlabeled, study-specific datasets
without requiring cell type labels. Neither multi-model support nor the
self-supervised path are shipped yet. This milestone makes the claim credible.

## Current state

- current foundation-model support is scGPT only
- generic PEFT support exists on the scGPT annotation path for:
  - full fine-tuning
  - LoRA
  - adapters
  - prefix tuning
  - IA3
- cross-model parity for those methods does not exist yet
- self-supervised adaptation path (masked gene modeling, contrastive objective)
  does not exist for any model

## Exit artifacts

- model-agnostic adapter layer: swapping scGPT for Geneformer or scFoundation
  does not require rewriting training code
- wrapper requirements for scFoundation, CellFM, and Nicheformer
- self-supervised adaptation path: takes unlabeled AnnData, adapts via masked
  gene modeling or contrastive objective without requiring cell type labels
- parity matrix for model capabilities
- parity matrix for PEFT methods across models
- unified checkpoint saving/loading that is strategy-agnostic
- benchmark readiness tracker across models

## Checklist

- [ ] define the common wrapper contract (model-agnostic adapter layer)
- [ ] define the self-supervised adaptation path specification (masked gene
  modeling or contrastive objective)
- [ ] define scFoundation integration requirements
- [ ] define CellFM integration requirements
- [ ] define Nicheformer integration requirements
- [ ] define LoRA parity requirements across models
- [ ] define adapter parity requirements across models
- [ ] define prefix-tuning parity requirements across models
- [ ] define IA3 parity requirements across models
- [ ] define supervised path parity (labeled AnnData, classification/regression
  target) across models
- [ ] define self-supervised path parity across models
- [ ] define parity criteria across models and methods
- [ ] define unified checkpoint saving and loading (strategy-agnostic)

## Risks / blockers

- the paper target can overpromise if model breadth is described before wrapper
  parity exists
- the self-supervised path requires a training objective not yet implemented
  on any model path
- Nicheformer has spatial-specific inputs that may not fit the gene-expression
  adapter contract cleanly

## Dependencies

- annotation pillar (M1)
- representation transfer pillar (M2)
- perturbation pillar (M3)
- spatial pillar (M4)

## Acceptance criteria

- the model-agnostic wrapper contract is defined and at least one non-scGPT
  model is integrated under it
- the self-supervised adaptation path is defined with a concrete training
  objective and at least one model implementation
- the model and PEFT expansion path is tracked explicitly in a parity matrix
  rather than as an implied promise
