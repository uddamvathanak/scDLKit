# Implementation map

This file is intentionally blunt. It separates the paper target from the actual
repo state.

## Models x status

| Model | Wrapper support | Inference | Frozen embeddings | Full fine-tuning | LoRA | Adapters | Prefix tuning | IA3 | Tests | Tutorials | Benchmarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scGPT | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Implemented | Pilot | Pilot |
| scFoundation | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| CellFM | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Nicheformer | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |

## Tasks x status

| Task | Dataset registry | Baseline metrics | Foundation-model path | PEFT comparison | Low-label regime | Cross-study regime | Tutorial | Guide | Figure readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Annotation | Implemented | Implemented | Pilot | Pilot | Implemented | Implemented | Implemented | Implemented | Pilot |
| Integration | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Perturbation | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Spatial | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |

## Paper figures x source artifacts

| Figure | Source script | Source datasets | Source metrics | Status |
| --- | --- | --- | --- | --- |
| Main performance figure | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | macro F1, balanced accuracy, accuracy | Pilot |
| Efficiency-performance Pareto figure | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | performance versus runtime and trainable parameter fraction | Pilot |
| Low-label curves | `scripts/run_annotation_benchmark.py` | PBMC plus human pancreas | task-specific performance across `1%`, `5%`, and `10%` labels | Pilot |
| Cross-study generalization figure | `scripts/run_annotation_benchmark.py` | OpenProblems human pancreas | macro F1 and balanced accuracy across held-out folds | Pilot |
| Task-specific qualitative figure | tutorial artifact bundles | human-pancreas annotation tutorial outputs | confusion matrix and embedding UMAPs | Implemented |
