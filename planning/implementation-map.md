# Implementation map

This file is intentionally blunt. It separates the paper target from the actual
repo state.

## Abstract pillar gaps

The abstract commits scDLKit to four pillars. Current gaps:

| Pillar | Claim | Current gap |
| --- | --- | --- |
| 1 — Unified adaptation interface | single training interface across multiple foundation models and PEFT strategies, including self-supervised path | multi-model support and self-supervised path not yet shipped |
| 2 — Distribution shift benchmarking | three formal evaluation regimes (full-label, low-label, cross-study) with measurable distribution shift as a covariate | regimes are defined in M1 checklist but not yet a formal cross-task benchmark object |
| 3 — Reproducible benchmark suite | four downstream tasks with standardized datasets, splits, and metrics | annotation is Pilot; representation transfer, perturbation, and spatial not yet shipped |
| 4 — Efficiency-aware evaluation | every benchmark run records trainable params, total params, runtime, peak memory, checkpoint size alongside performance | efficiency metrics not yet systematically recorded in benchmark output |

## Milestone sequence (per abstract dev order)

| Milestone | Focus | Pillars | Status |
| --- | --- | --- | --- |
| M0 | Publication OS | — | complete |
| M1 | Annotation pillar | Pillar 2 (regimes), Pillar 3 (annotation), Pillar 4 (efficiency recording) | blocked |
| M2 | Representation transfer pillar | Pillar 2, Pillar 3 (representation transfer), Pillar 4 | planned |
| M3 | Perturbation pillar | Pillar 2, Pillar 3 (perturbation), Pillar 4 | planned |
| M4 | Spatial pillar | Pillar 2, Pillar 3 (spatial), Pillar 4 | planned |
| M5 | Cross-model PEFT expansion + self-supervised path | Pillar 1 | planned |
| M6 | Paper assets and manuscript | — | planned |

Note: spatial is deferred from its previous M2 position because the abstract's
recommended dev order prioritizes representation transfer before expanding to
spatial and perturbation.

## Models x status

M1 note:
- annotation fine-tuning is implemented only on `scGPT`
- cross-model parity for `scFoundation`, `CellFM`, and `Nicheformer` is M5 work
- self-supervised adaptation path is M5 work

| Model | Wrapper support | Inference | Frozen embeddings | Full fine-tuning | LoRA | Adapters | Prefix tuning | IA3 | Self-supervised path | Tests | Tutorials | Benchmarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scGPT | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Planned | Implemented | Pilot | Pilot |
| scFoundation | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| CellFM | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Nicheformer | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |

## Tasks x status

| Task | Dataset registry | Baseline metrics | Efficiency metrics | Foundation-model path | PEFT comparison | Low-label regime | Cross-study regime | Tutorial | Guide | Figure readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Annotation | Implemented | Implemented | Planned | Pilot (`scGPT` only) | Pilot (`scGPT` only) | Implemented | Implemented | Implemented | Implemented | Pilot |
| Representation transfer | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Perturbation | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Spatial | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |

## Paper figures x source artifacts

| Figure | Source script | Source datasets | Source metrics | Status |
| --- | --- | --- | --- | --- |
| Main performance figure | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | macro F1, balanced accuracy, accuracy | Pilot |
| Efficiency-performance Pareto figure | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | performance vs. runtime, trainable param fraction, peak memory | Pilot |
| Low-label curves | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | task-specific performance across `1%`, `5%`, and `10%` labels | Pilot |
| Cross-study generalization figure | `scripts/run_annotation_benchmark.py` | OpenProblems human pancreas | macro F1 and balanced accuracy across held-out folds | Pilot |
| Task-specific qualitative figure | tutorial artifact bundles | human-pancreas annotation tutorial outputs | confusion matrix and embedding UMAPs | Implemented |
