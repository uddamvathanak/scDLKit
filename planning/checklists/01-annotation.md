# Milestone 1: annotation pillar

Status: blocked

## Objective

Turn annotation into the first paper-ready task pillar.

## Why this matters for the paper

Annotation is the strongest implemented research-facing path in the current
repo. It is the fastest place to prove the benchmark, adaptation, and artifact
story before broadening to spatial, integration, and perturbation.

## Current state

- the annotation task spec is now encoded in repo-tracked benchmark dataclasses
- the generic PEFT layer exists and is wired into the scGPT annotation path
- Milestone 1 model breadth is still intentionally `scGPT` only
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
- the remaining work is:
  - benchmark artifact freeze
  - explicit review of the remaining annotation-pillar risks
  - public promotion from `Pilot` to `Implemented` only if the reviewed
    evidence bundle supports it
- the first full-profile benchmark attempt on `2026-03-24` did not complete
  within the local 4-hour runtime budget and currently blocks milestone closure

## Exit artifacts

- annotation task spec
- dataset shortlist and regime registry
- benchmark matrix across frozen, full-FT, LoRA, adapters, prefix, and IA3
- main annotation tutorial designation
- figure-ready artifact list
- first full benchmark artifact bundle with efficiency metrics per run
- Pareto figure (performance vs. trainable param fraction and runtime)

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
- [x] lock Milestone 1 model breadth to `scGPT` and carry cross-model PEFT parity into Milestone 5
- [x] install the dedicated annotation benchmark workflow
- [x] add efficiency metric recording to benchmark run output (trainable param
  count, total param count, wall-clock training time, peak memory, checkpoint
  size)
- [ ] make the benchmark execution path feasible for the full matrix within the
  runtime budget
- [ ] run and review the first full benchmark artifact bundle
- [ ] verify the Pareto figure (performance vs. trainable param fraction and
  runtime) is generated as part of the artifact bundle
- [ ] review the remaining annotation-pillar risks and either close them or carry them forward explicitly

## Risks / blockers

- the current foundation fine-tuning path is still scGPT-only
- cross-model parity is not solved in Milestone 1 and remains open work for
  Milestone 5
- the full annotation matrix is currently too heavy for the present execution
  path and timeout budget to complete the first full artifact freeze
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
- annotation milestone leaves no ambiguity that current fine-tuning support is
  `scGPT` only and that cross-model breadth is deferred to Milestone 5
- annotation remains `Pilot` until the benchmark artifact bundle is reviewed
  and can be promoted using explicit evidence rather than impression
- every benchmark run artifact includes efficiency metrics alongside performance
  metrics (Pillar 4 requirement, sets the standard for all subsequent milestones)
- the three evaluation regimes (full-label, low-label, cross-study) are
  formalized as reusable benchmark objects, not just as annotation-specific
  conventions
- remaining risks are either resolved for the annotation pillar or explicitly
  carried forward in the planning files before the checklist is closed
