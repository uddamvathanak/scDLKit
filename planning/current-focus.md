# Current focus

## Paper framing

The abstract commits scDLKit to four concrete pillars:

- Pillar 1 - Unified adaptation interface: single training interface across
  multiple foundation models and multiple PEFT strategies, including a
  self-supervised adaptation path for unlabeled data
- Pillar 2 - Distribution shift benchmarking: three formal evaluation regimes
  (full-label, low-label, cross-study) with measurable shift between train and
  eval
- Pillar 3 - Reproducible benchmark suite: four downstream tasks (annotation,
  representation transfer, perturbation, spatial) with standardized datasets,
  preprocessing, splits, and metrics
- Pillar 4 - Efficiency-aware evaluation: every benchmark run records trainable
  parameter count, total parameter count, wall-clock training time, peak memory,
  and checkpoint size alongside performance metrics

## Target journal

- `Bioinformatics` Original Paper
- planning-only target for the eventual full-toolkit paper
- current repo state is not submission-ready for that paper; M2-M5 still need
  to land before the manuscript is defensible
- milestone work is now judged against whether it contributes evidence that can
  survive a `Bioinformatics` paper, not just whether it ships a feature

## Milestone sequence

The recommended development order from the abstract specification:

1. formalize evaluation regimes as reusable benchmark objects across tasks
2. use annotation as the first validated paper pillar and start representation
   transfer
3. propagate efficiency recording to every downstream benchmark run
4. ship perturbation and spatial task modules
5. add multi-model support and a self-supervised adaptation path

Milestone numbers:

- M0: Publication OS - complete
- M1: Annotation pillar - complete
- M2: Representation transfer pillar - planned
- M3: Perturbation pillar - planned
- M4: Spatial pillar - planned
- M5: Cross-model PEFT expansion + self-supervised adaptation path - planned
- M6: Paper assets and manuscript - planned

## Current objective

Current milestone:

- M2: Representation transfer pillar

Why now:

- M1 is the first validated paper pillar and now sets the benchmark pattern
- representation transfer is the next missing task pillar in the full-toolkit
  `Bioinformatics` paper target
- the current k-fold rerun and figure review are M1 hardening work, not a
  reason to keep M1 marked blocked

Top 3 deliverables:

1. define the representation transfer task contract, dataset shortlist, and
   regime registry
2. inherit the M1 benchmark conventions: held-out shift regimes, k-fold
   evaluation, and efficiency recording
3. scope M2 outputs to fit the `Bioinformatics` manuscript target rather than a
   standalone feature demo

Blockers:

- none at the planning level; current risk is execution discipline rather than
  missing milestone definition

Exit criteria:

- do not describe the full-toolkit paper as submission-ready before M2-M5 exist
- do not reopen M1 status unless the reviewed evidence bundle regresses

Next milestone after this one:

- M3: Perturbation pillar
