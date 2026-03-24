# Annotation Benchmarks

Status: experimental.

This guide summarizes how scDLKit currently frames annotation adaptation evidence.

The goal is not to claim that a foundation model always wins. The goal is to
give researchers a reproducible way to compare:

- `PCA + logistic regression`
- frozen scGPT probe
- head-only tuning
- full fine-tuning
- LoRA tuning
- adapter tuning
- prefix tuning
- IA3 tuning

on more than one labeled human dataset while keeping the outputs Scanpy-friendly.

## Datasets in the current evidence story

- `pbmc68k_reduced`
- OpenProblems human pancreas

These are both human scRNA-seq datasets, which matches the current experimental
scGPT `whole-human` scope.

## Regimes in the annotation pillar

- full-label benchmark on `pbmc68k_reduced` and human pancreas
- low-label benchmark on both datasets at `1%`, `5%`, and `10%`
- cross-study benchmark on human pancreas through held-out technology families

The main public tutorial stays on the `quickstart` pancreas subset and compares
only `frozen_probe` and `head`. The heavier regime matrix belongs to the
dedicated benchmark workflow, not the default docs notebook path.

## What the current comparisons are meant to answer

- Is the frozen checkpoint already good enough for the label space?
- Does a small trainable head recover the gap without a heavier adaptation step?
- Does full fine-tuning materially improve on lighter strategies?
- Which PEFT method is the best cost-performance tradeoff on this dataset?
- Can the best adapted runner be saved and reused later?

## What to conclude from the benchmark outputs

- Sometimes the frozen probe is already strong enough.
- Sometimes head-only tuning is the best cost-versus-performance tradeoff.
- Full fine-tuning is the unconstrained reference point, not the default
  recommendation.
- LoRA, adapters, prefix tuning, and IA3 are available when more expressive
  adaptation is worth the added runtime and configuration complexity.
- The main value of scDLKit is not a universal leaderboard claim. It is the
  ability to make this comparison reproducible and inspectable on `AnnData`.

## Why scDLKit is useful here

- it keeps the workflow `AnnData`-native
- it writes embeddings and predictions back into Scanpy-compatible structures
- it saves standard comparison artifacts instead of hiding the result behind one score
- it can save and reload the best adapted runner for later reuse
- it now exposes a dedicated annotation benchmark runner and workflow for the
  heavier matrix, separate from the beginner tutorial path

## Artifact and ranking rules

The annotation pillar ranks strategies by:

1. `macro_f1`
2. `balanced_accuracy`
3. `accuracy`
4. lower `trainable_parameters`
5. lower `runtime_sec`

The benchmark artifact bundle is shaped for paper figures:

- full-label summary across PBMC and pancreas
- low-label curves for `1%`, `5%`, and `10%`
- pancreas cross-study held-out-batch results
- efficiency-performance Pareto plots
- qualitative UMAP and confusion-matrix outputs from the main tutorial path

Use `scripts/run_annotation_benchmark.py` for the full matrix and
`.github/workflows/annotation-benchmark.yml` for the scheduled or manual heavy
run.

## Recommended reading order

1. [Main annotation tutorial: human-pancreas wrapper workflow](/_tutorials/scgpt_human_pancreas_annotation)
2. [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)
3. [Experimental annotation quickstart API](../api/annotation.md)
4. [Experimental foundation helpers](../api/foundation.md)
5. [Tutorial execution status](/tutorials/status)

## Notes and caveats

- This is still an experimental feature line.
- The current public checkpoint scope is still only scGPT `whole-human`, so the
  annotation pillar is implemented on `scGPT` only.
- This guide does not claim non-human, multimodal, perturbation, or spatial support.
- The beyond-PBMC evidence path is intentionally heavier than the standard PR CI path.
