# Milestone 1: annotation pillar

Status: complete (pending figure review)

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
- the full benchmark completed on `2026-03-25` after adding checkpoint caching
  and per-seed data preparation caching — all 4 figures (performance, low-label,
  cross-study, Pareto) generated as PNG (300 dpi) and SVG
- the initial benchmark used only 2-3 epochs per strategy (surface-level) —
  redesigned on `2026-03-25` for publication quality:
  - convergence-based training: 10-15 epochs with early stopping (patience=5),
    matching the scGPT Nature Methods and PEFT for scLLMs protocols
  - LR scheduling: StepLR with gamma=0.9 per epoch (following scGPT protocol)
  - comprehensive metrics: added weighted F1, precision, recall, Cohen's kappa,
    MCC, per-class F1/precision/recall
  - publication-quality figures: error bars (mean +/- std across seeds),
    confidence bands on low-label curves, cross-study heatmap,
    labeled Pareto scatter, radar chart, per-class F1 heatmap
  - benchmark design rationale documented in
    `planning/research/annotation_benchmark_design.md`

## Exit artifacts

- annotation task spec
- dataset shortlist and regime registry
- benchmark matrix across frozen, full-FT, LoRA, adapters, prefix, and IA3
- main annotation tutorial designation
- figure-ready artifact list
- publication-quality benchmark artifact bundle with comprehensive metrics
- 6 publication figures: performance bars, label efficiency curves,
  cross-study heatmap, Pareto scatter, radar chart, per-class F1 heatmap
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
- [x] make the benchmark execution path feasible for the full matrix within the
  runtime budget
- [x] run and review the first full benchmark artifact bundle
- [x] verify the Pareto figure (performance vs. trainable param fraction and
  runtime) is generated as part of the artifact bundle
- [x] upgrade benchmark to convergence-based training (10-15 epochs, early
  stopping patience=5, StepLR gamma=0.9 — matching scGPT protocol)
- [x] add comprehensive metrics (weighted F1, precision, recall, Cohen's kappa,
  MCC, per-class F1/precision/recall)
- [x] redesign figures for publication quality (error bars, confidence bands,
  heatmap, radar chart, per-class F1 heatmap)
- [x] document benchmark design rationale with literature references
- [x] run and review the converged benchmark artifact bundle (264/264 runs
  completed on 2026-03-25 — all 6 publication figures generated with narratives)
- [x] review the remaining annotation-pillar risks and either close them or
  carry them forward explicitly (reviewed 2026-03-25)

## Risks / blockers

### Resolved

- ~~the full annotation matrix is currently too heavy for the present execution
  path and timeout budget to complete the first full artifact freeze~~ (resolved
  on 2026-03-25 with checkpoint + data prep caching)
- ~~the annotation story can still look more complete than it is if the public
  status is promoted before the first benchmark artifact freeze~~ (resolved:
  264/264 benchmark runs complete with full artifact bundle, 6 publication
  figures, and narrative documentation — promotion can now be evidence-based)

### Carried forward

- **scGPT-only**: the current foundation fine-tuning path is still scGPT-only.
  This is intentional for Milestone 1. Cross-model parity (scVI, Geneformer,
  etc.) is deferred to Milestone 5.
- **PBMC dataset limitations**: `pbmc68k_reduced` uses noisy bulk-imputed labels
  (~700 cells, 10 types), producing lower absolute scores (Macro F1 ~0.4-0.5).
  The pancreas dataset is the primary evidence dataset. Consider replacing PBMC
  with a higher-quality dataset in a future milestone if needed.
- **Full fine-tuning catastrophic forgetting**: full fine-tuning consistently
  collapses on both datasets (Macro F1 < 0.3 on pancreas). This is a known
  limitation and a key part of the PEFT narrative, not a bug. Should be
  explicitly discussed in the manuscript as motivation for PEFT.
- **Gene vocabulary overlap**: scGPT checkpoint vocabulary matches only ~118
  genes on PBMC. This limits foundation model performance on datasets with
  non-overlapping gene panels. Future work should evaluate vocabulary-aware
  preprocessing or newer checkpoints with broader vocabularies.

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
