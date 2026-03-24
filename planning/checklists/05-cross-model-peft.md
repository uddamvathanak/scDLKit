# Milestone 5: cross-model PEFT expansion

Status: planned

## Objective

Expand the model and PEFT axes toward the paper target without pretending
everything will land at the same maturity level.

## Why this matters for the paper

The paper claims a cross-model adaptation framework. That claim only becomes
credible when the wrapper, PEFT, and benchmark story exists beyond scGPT.

## Current state

- current foundation-model support is scGPT only
- generic PEFT support now exists on the scGPT annotation path for:
  - full fine-tuning
  - LoRA
  - adapters
  - prefix tuning
  - IA3
- cross-model parity for those methods does not exist yet

## Exit artifacts

- wrapper requirements for scFoundation, CellFM, and Nicheformer
- parity matrix for model capabilities
- parity matrix for PEFT methods
- benchmark readiness tracker across models

## Checklist

- [ ] define the common wrapper contract
- [ ] define scFoundation integration requirements
- [ ] define CellFM integration requirements
- [ ] define Nicheformer integration requirements
- [ ] define adapters integration requirements
- [ ] define prefix-tuning integration requirements
- [ ] define IA3 integration requirements
- [ ] define parity criteria across models and methods

## Risks / blockers

- the paper target can overpromise if model breadth is described before wrapper parity exists

## Dependencies

- annotation pillar
- spatial pillar
- integration pillar
- perturbation pillar

## Acceptance criteria

- the model and PEFT expansion path is tracked explicitly instead of as an implied promise
