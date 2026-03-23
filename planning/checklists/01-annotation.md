# Milestone 1: annotation pillar

Status: active

## Objective

Turn annotation into the first paper-ready task pillar.

## Why this matters for the paper

Annotation is the strongest implemented research-facing path in the current
repo. It is the fastest place to prove the benchmark, adaptation, and artifact
story before broadening to spatial, integration, and perturbation.

## Current state

- scGPT annotation adaptation exists
- LoRA exists as the current PEFT method
- wrapper-first annotation tutorials exist
- beyond-PBMC annotation evidence exists
- full fine-tuning, adapters, prefix tuning, IA3, low-label, and cross-study
  benchmarking are still missing

## Exit artifacts

- annotation task spec
- dataset shortlist and registry plan
- benchmark matrix across frozen, full-FT, LoRA, adapters, prefix, and IA3
- main annotation tutorial designation
- figure-ready artifact list

## Checklist

- [ ] lock the annotation task spec
- [ ] lock the annotation dataset shortlist
- [ ] define the full-label regime
- [ ] define the low-label regime
- [ ] define the cross-study regime
- [ ] define the frozen / linear-probe baseline
- [ ] define the full fine-tuning baseline
- [ ] define the LoRA baseline
- [ ] define the adapters baseline
- [ ] define the prefix-tuning baseline
- [ ] define the IA3 baseline
- [ ] decide which existing annotation notebook becomes the main research-facing tutorial
- [ ] define the annotation figure inventory

## Risks / blockers

- the current foundation path is still scGPT-only
- the current PEFT path is still LoRA-only
- the annotation story can look more complete than it is if full-FT and other PEFT baselines are not tracked explicitly

## Dependencies

- publication operating system
- current scGPT annotation benchmark and artifact scripts

## Acceptance criteria

- annotation milestone leaves no ambiguity about task spec, regimes, baselines, or main tutorial
- annotation is clearly marked as `Pilot` or `Implemented` based on explicit evidence rather than impression
