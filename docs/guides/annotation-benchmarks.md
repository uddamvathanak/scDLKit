# Annotation Benchmarks

Status: experimental.

This guide summarizes how scDLKit currently frames annotation adaptation evidence.

The goal is not to claim that a foundation model always wins. The goal is to
give researchers a reproducible way to compare:

- `PCA + logistic regression`
- frozen scGPT probe
- head-only tuning
- LoRA tuning

on more than one labeled human dataset while keeping the outputs Scanpy-friendly.

## Datasets in the current evidence story

- `pbmc68k_reduced`
- OpenProblems human pancreas

These are both human scRNA-seq datasets, which matches the current experimental
scGPT `whole-human` scope.

## What the current comparisons are meant to answer

- Is the frozen checkpoint already good enough for the label space?
- Does a small trainable head recover the gap without a heavier adaptation step?
- Is LoRA worth the extra runtime on this dataset?
- Can the best adapted runner be saved and reused later?

## What to conclude from the benchmark outputs

- Sometimes the frozen probe is already strong enough.
- Sometimes head-only tuning is the best cost-versus-performance tradeoff.
- LoRA is available when extra adaptation is worth the added runtime.
- The main value of scDLKit is not a universal leaderboard claim. It is the
  ability to make this comparison reproducible and inspectable on `AnnData`.

## Why scDLKit is useful here

- it keeps the workflow `AnnData`-native
- it writes embeddings and predictions back into Scanpy-compatible structures
- it saves standard comparison artifacts instead of hiding the result behind one score
- it can save and reload the best adapted runner for later reuse

## Recommended reading order

1. [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)
2. [Experimental scGPT human-pancreas annotation](/_tutorials/scgpt_human_pancreas_annotation)
3. [Experimental annotation quickstart API](../api/annotation.md)
4. [Experimental foundation helpers](../api/foundation.md)

## Notes and caveats

- This is still an experimental feature line.
- The current public checkpoint scope is still only scGPT `whole-human`.
- This guide does not claim non-human, multimodal, perturbation, or spatial support.
- The beyond-PBMC evidence path is intentionally heavier than the standard PR CI path.
