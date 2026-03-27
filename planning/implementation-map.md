# Implementation map

This file is intentionally blunt. It separates the paper target from the actual
repo state.

## Target journal

- `Bioinformatics` Original Paper
- planning-only target for the eventual full-toolkit paper
- the chosen paper scope is broader than the current implementation truth
- a defensible submission still requires M2-M4 task pillars plus M5
  cross-model and self-supervised scope

Current evidence already supports an annotation-scoped software and benchmark
story. The chosen manuscript target is larger: a full-toolkit paper that
connects all four downstream tasks through shared benchmark regimes,
efficiency-aware reporting, and a broader adaptation story than M1 alone can
justify.

## Abstract pillar gaps

The abstract commits scDLKit to four pillars. Current gaps:

| Pillar | Claim | Current gap |
| --- | --- | --- |
| 1 - Unified adaptation interface | single training interface across multiple foundation models and PEFT strategies, including self-supervised path | multi-model support and self-supervised path are not yet shipped |
| 2 - Distribution shift benchmarking | three formal evaluation regimes (full-label, low-label, cross-study) with measurable distribution shift as a covariate | annotation now has locked regimes and k-fold evaluation, but the regime language is not yet promoted into a reusable cross-task benchmark object |
| 3 - Reproducible benchmark suite | four downstream tasks with standardized datasets, splits, and metrics | annotation is the first completed pillar; representation transfer, perturbation, and spatial are not yet shipped |
| 4 - Efficiency-aware evaluation | every benchmark run records trainable params, total params, runtime, peak memory, checkpoint size alongside performance | efficiency metrics are now recorded for annotation runs, but not yet standardized across the future task runners |

## Milestone sequence (per abstract dev order)

| Milestone | Focus | Pillars | Status |
| --- | --- | --- | --- |
| M0 | Publication OS | - | complete |
| M1 | Annotation pillar | Pillar 2 (regimes), Pillar 3 (annotation), Pillar 4 (efficiency recording) | complete |
| M2 | Representation transfer pillar | Pillar 2, Pillar 3 (representation transfer), Pillar 4 | planned |
| M3 | Perturbation pillar | Pillar 2, Pillar 3 (perturbation), Pillar 4 | planned |
| M4 | Spatial pillar | Pillar 2, Pillar 3 (spatial), Pillar 4 | planned |
| M5 | Cross-model PEFT expansion + self-supervised path | Pillar 1 | planned |
| M6 | Paper assets and manuscript | - | planned |

Note: spatial remains deferred from its previous M2 position because the
abstract's recommended development order prioritizes representation transfer
before expanding to spatial and perturbation.

## Models x status

M1 note:

- annotation fine-tuning is implemented only on `scGPT`
- cross-model parity for `scFoundation`, `CellFM`, and `Nicheformer` is M5 work
- self-supervised adaptation path is M5 work

| Model | Wrapper support | Inference | Frozen embeddings | Full fine-tuning | LoRA | Adapters | Prefix tuning | IA3 | Self-supervised path | Tests | Tutorials | Benchmarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scGPT | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Planned | Implemented | Implemented | Implemented |
| scFoundation | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| CellFM | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Nicheformer | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |

## Tasks x status

| Task | Dataset registry | Baseline metrics | Efficiency metrics | Foundation-model path | PEFT comparison | Low-label regime | Cross-study regime | Tutorial | Guide | Figure readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Annotation | Implemented | Implemented | Implemented | Implemented (`scGPT` only) | Implemented (`scGPT` only) | Implemented | Implemented | Implemented | Implemented | Implemented |
| Representation transfer | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Perturbation | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Spatial | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |

## Paper figures x source artifacts

| Figure | Source script | Source datasets | Source metrics | Status |
| --- | --- | --- | --- | --- |
| Main performance figure | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | macro F1, balanced accuracy, accuracy, weighted F1 | Implemented |
| Label-efficiency curves | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | task performance across low-label fractions with fold-level variance | Implemented |
| Cross-study generalization heatmap | `scripts/run_annotation_benchmark.py` | OpenProblems human pancreas | macro F1 and balanced accuracy across held-out folds | Implemented |
| Efficiency-performance Pareto figure | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | performance vs. runtime and trainable parameter fraction | Implemented |
| Radar comparison figure | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | normalized multi-metric comparison across strategies | Implemented |
| Per-class F1 heatmap | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | per-class F1 across strategies | Implemented |
| Task-specific qualitative figure | tutorial artifact bundles | human-pancreas annotation tutorial outputs | confusion matrix and embedding UMAPs | Implemented |
