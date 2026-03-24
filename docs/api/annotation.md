# Experimental Annotation Quickstart API

## What it is

Status: experimental.

This page documents the easiest public path for labeled annotation adaptation:

- `adapt_annotation(...)` for the one-call workflow
- `inspect_annotation_data(...)` for preflight checks
- `AnnotationRunner` for the explicit inspect-fit-predict-annotate-save-load flow

The current implementation routes only to the experimental scGPT `whole-human`
annotation path for human scRNA-seq data.

## When to use it

Use this page instead of [TaskRunner](./taskrunner.md) when:

- you already have labels in `adata.obs`
- your goal is annotation adaptation, not just a baseline embedding
- you want predictions and embeddings written back into `AnnData`
- you want to compare frozen and tuned strategies with minimal code

Use [Experimental foundation helpers](./foundation.md) when you want the lower-level scGPT-specific route underneath this alias layer.

## Minimal example

```python
from scdlkit import adapt_annotation

runner = adapt_annotation(
    adata,
    label_key="cell_type",
    output_dir="artifacts/scgpt_annotation",
)

runner.annotate_adata(adata, obs_key="scgpt_label", embedding_key="X_scgpt_best")
runner.save("artifacts/scgpt_annotation/best_model")
```

## Parameters

- `label_key`: required `adata.obs` column containing the target annotation labels.
- `checkpoint`: currently fixed to the experimental scGPT `whole-human` checkpoint.
- `strategies`: default quickstart is `("frozen_probe", "head")`. Additional
  opt-in strategy names are `"full_finetune"`, `"lora"`, `"adapter"`,
  `"prefix_tuning"`, and `"ia3"`.
- `strategy_configs`: optional per-strategy config mapping for the heavier PEFT
  comparison surface exposed under `scdlkit.foundation`.
- `lora_config`: backward-compatible alias for
  `strategy_configs={"lora": LoRAConfig(...)}` in the `0.1.x` line.
- `batch_size`, `val_size`, `test_size`, `random_state`, `device`: wrapper
  training and split defaults.
- `output_dir`: optional artifact directory for reports, plots, and saved runner state.

## Input expectations

- input must be human scRNA-seq stored in `anndata.AnnData`.
- `label_key` must exist in `adata.obs` and contain at least two label categories for training.
- the expression matrix must be non-negative for the scGPT tokenization path.
- gene overlap with the `whole-human` vocabulary must be sufficient; `inspect_annotation_data(...)` exposes that check before fitting.
- optional batch or study metadata can stay in `adata.obs` and will be carried
  through for downstream reporting when present.

## Returns / outputs

- `inspect_annotation_data(...)` returns a `ScGPTAnnotationDataReport`.
- `adapt_annotation(...)` returns a fitted `AnnotationRunner`.
- `AnnotationRunner.predict(...)` returns `label_codes`, `labels`, `probabilities`, and `latent`.
- `AnnotationRunner.annotate_adata(...)` writes labels to `adata.obs` and embeddings to `adata.obsm`.
- `AnnotationRunner.save(...)` writes a directory with `manifest.json` and `model_state.pt`.
- strategy comparison artifacts include per-strategy metrics with
  `macro_f1`, `accuracy`, `balanced_accuracy`, and multiclass
  `auroc_ovr` when probability outputs make it valid.

## Failure modes / raises

- `ImportError` if the package was installed without the `foundation` extra.
- `ValueError` if labels are missing, class counts are too small, or gene overlap is insufficient.
- `ValueError` if an unsupported strategy name is requested or a strategy config
  does not match the selected strategy.
- `ValueError` if both `strategy_configs` and `lora_config` are supplied.
- `RuntimeError` if you try to predict, annotate, or save before fitting or loading a runner.
- `ValueError` if the saved runner manifest is incomplete or incompatible.

## Notes / caveats

- This surface is experimental even though the aliases live at `scdlkit`.
- The beginner default is intentionally CPU-friendly: frozen probe plus head-only tuning.
- The heavier annotation benchmark surface extends to:
  - full fine-tuning
  - LoRA
  - adapters
  - prefix tuning
  - IA3
- The current public model implementation is still `scGPT` only.
- `TaskRunner` is not extended for this path in the current release line.

## Related tutorial(s)

- [Main annotation tutorial: human-pancreas wrapper workflow](/_tutorials/scgpt_human_pancreas_annotation)
- [Experimental scGPT dataset-specific annotation](/_tutorials/scgpt_dataset_specific_annotation)
- [Experimental scGPT cell-type annotation](/_tutorials/scgpt_cell_type_annotation)
- [Tutorial execution status](/tutorials/status)

```{eval-rst}
.. autoclass:: scdlkit.AnnotationRunner
   :members:
   :inherited-members:
```

```{eval-rst}
.. autofunction:: scdlkit.inspect_annotation_data
```

```{eval-rst}
.. autofunction:: scdlkit.adapt_annotation
```
