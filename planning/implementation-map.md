# Implementation map

This file is intentionally blunt. It separates the paper target from the actual
repo state.

## Models x status

| Model | Wrapper support | Inference | Frozen embeddings | Full fine-tuning | LoRA | Adapters | Prefix tuning | IA3 | Tests | Tutorials | Benchmarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scGPT | Pilot | Implemented | Implemented | Planned | Pilot | Planned | Planned | Planned | Implemented | Implemented | Pilot |
| scFoundation | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| CellFM | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Nicheformer | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |

## Tasks x status

| Task | Dataset registry | Baseline metrics | Foundation-model path | PEFT comparison | Low-label regime | Cross-study regime | Tutorial | Guide | Figure readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Annotation | Pilot | Pilot | Pilot | Pilot | Planned | Planned | Pilot | Pilot | Pilot |
| Integration | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Perturbation | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |
| Spatial | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned | Planned |

## Paper figures x source artifacts

| Figure | Source script | Source datasets | Source metrics | Status |
| --- | --- | --- | --- | --- |
| Main performance figure | `scripts/run_external_annotation_evidence.py` | PBMC plus human pancreas | accuracy, macro F1, runtime, trainable parameters | Pilot |
| Efficiency-performance Pareto figure | `scripts/run_external_annotation_evidence.py` plus future summary script | PBMC plus human pancreas | performance versus runtime or trainable parameter fraction | Planned |
| Low-label curves | not yet assigned | not yet assigned | task-specific performance across label fractions | Planned |
| Cross-study generalization figure | not yet assigned | not yet assigned | task-specific cross-study metrics | Planned |
| Task-specific qualitative figure | tutorial artifact bundles | current annotation UMAP and confusion matrix artifacts | qualitative task outputs | Pilot |
