# Milestone 1 review

Date:
- 2026-03-24

Milestone:
- Annotation pillar

## Benchmark run review

Run location:
- local GPU environment

Environment:
- `conda` env: `scdlkit`
- CUDA device visible: `NVIDIA GeForce RTX 3080 Laptop GPU`

Primary command:

```bash
conda run -n scdlkit python scripts/run_annotation_benchmark.py --profile full --output-dir artifacts/annotation_pillar
```

Outcome:
- `fail`

Failure mode:
- the run did not complete within the local 4-hour execution budget
- command timeout: `14404` seconds

Observed partial progress:
- completed per-run artifacts: `22`
- expected per-run artifacts for the full matrix: `264`
- progress reached only part of `full_label / pbmc68k_reduced`

Bundle-shape result:
- `artifacts/annotation_pillar/runs/`: present
- `artifacts/annotation_pillar/full_label/strategy_metrics.csv`: missing
- `artifacts/annotation_pillar/low_label/strategy_metrics.csv`: missing
- `artifacts/annotation_pillar/cross_study/strategy_metrics.csv`: missing
- `artifacts/annotation_pillar/figures/annotation_performance.png`: missing
- `artifacts/annotation_pillar/summary.md`: missing
- `artifacts/annotation_pillar/summary.json`: missing

Interpretation:
- this is not a logic failure in the benchmark script itself
- it is an execution-path blocker at the current Milestone 1 matrix size and runtime budget
- the existing GitHub workflow timeout of `240` minutes on `ubuntu-latest` CPU is unlikely to be sufficient, because the local GPU run did not finish within the same time budget

## Tutorial review

Tutorial command:

```bash
conda run -n scdlkit python scripts/prepare_tutorial_notebooks.py --execute published --only scgpt_human_pancreas_annotation.ipynb --skip-assets
python scripts/render_tutorial_status.py
```

Outcome:
- `pass`

Verified state:
- the main human-pancreas annotation tutorial executed successfully
- the published tutorial remains a static executed tutorial
- artifact check remains `passed`
- the tutorial status page rendered successfully

## Figure review

Chosen paper-facing figure:
- `figures/annotation_performance.png`

Current state:
- not generated, because the full benchmark bundle did not complete

## Risk review

### Risk 1: scGPT-only fine-tuning

Outcome:
- `Accepted limitation for Milestone 1`

Reason:
- Milestone 1 was explicitly scoped to `scGPT`
- cross-model parity remains Milestone 5 work

### Risk 2: heavy benchmark runtime and cache sensitivity

Outcome:
- `Carried forward within Milestone 1`

Reason:
- this is the current blocking issue for milestone closure
- the full benchmark bundle does not yet complete within the current execution path and timeout budget

### Risk 3: public story overstating readiness

Outcome:
- `Resolved only if annotation remains Pilot`

Rule:
- do not promote annotation publicly from `Pilot` to `Implemented`
- do not mark Milestone 1 complete while the benchmark bundle remains incomplete

## Closure decision

Milestone 1 status:
- `blocked`

Reason:
- the full benchmark artifact bundle has not been generated and reviewed

What is complete:
- annotation code path
- benchmark runner and workflow scaffolding
- main tutorial execution path
- planning and public status framing

What is not complete:
- full benchmark artifact freeze
- paper-facing main performance figure
- milestone closure

## Recommended next fix

- make the annotation benchmark execution path resumable or otherwise feasible for the full matrix
- then rerun the full benchmark bundle and repeat this review
