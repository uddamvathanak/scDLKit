# Milestone 1: annotation pillar

Status: active

## Objective

Turn annotation into the first paper-ready task pillar.

## Why this matters for the paper

Annotation is the strongest implemented research-facing path in the current
repo. It is the fastest place to prove the benchmark, adaptation, and artifact
story before broadening to spatial, integration, and perturbation.

## Current state

- the annotation task spec is now encoded in repo-tracked benchmark dataclasses
- the generic PEFT layer exists and is wired into the scGPT annotation path
- scGPT annotation now supports:
  - `frozen_probe`
  - `head`
  - `full_finetune`
  - `lora`
  - `adapter`
  - `prefix_tuning`
  - `ia3`
- the human-pancreas notebook is the main published annotation tutorial and is
  prepared as a static docs notebook with visible status metadata
- the dedicated annotation benchmark runner and workflow now exist
- the remaining work is benchmark artifact freeze and public promotion from
  `Pilot` to `Implemented`

## Exit artifacts

- annotation task spec
- dataset shortlist and regime registry
- benchmark matrix across frozen, full-FT, LoRA, adapters, prefix, and IA3
- main annotation tutorial designation
- figure-ready artifact list
- first full benchmark artifact bundle

## Checklist

- [x] lock the annotation task spec
- [x] lock the annotation dataset shortlist
- [x] define the full-label regime
- [x] define the low-label regime
- [x] define the cross-study regime
- [x] define the frozen / linear-probe baseline
- [x] define the full fine-tuning baseline
- [x] define the LoRA baseline
- [x] define the adapters baseline
- [x] define the prefix-tuning baseline
- [x] define the IA3 baseline
- [x] decide which existing annotation notebook becomes the main research-facing tutorial
- [x] define the annotation figure inventory
- [x] install the dedicated annotation benchmark workflow
- [ ] run and review the first full benchmark artifact bundle

## Risks / blockers

- the current foundation path is still scGPT-only
- the full annotation matrix is heavy enough that cache misses or dataset
  download failures can distort validation time
- the annotation story can still look more complete than it is if the public
  status is promoted before the first benchmark artifact freeze

## Dependencies

- publication operating system
- current scGPT annotation benchmark and artifact scripts
- OpenProblems pancreas cache path
- tutorial publication status helpers

## Acceptance criteria

- annotation milestone leaves no ambiguity about task spec, regimes, baselines,
  or main tutorial
- annotation remains `Pilot` until the benchmark artifact bundle is reviewed
  and can be promoted using explicit evidence rather than impression
