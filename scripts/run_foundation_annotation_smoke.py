"""Run the annotation benchmark path once and reuse its metrics in CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_quality_suite import (  # noqa: E402
    DATASET_SPECS,
    PROFILE_DEFAULTS,
    _load_dataset,
    _subset_adata_for_foundation,
    _subset_foundation_genes,
    run_scgpt_annotation_strategy,
)

from scdlkit import AnnotationRunner, adapt_annotation, inspect_annotation_data  # noqa: E402
from scdlkit.foundation import ScGPTAnnotationRunner  # noqa: E402


def _relative_artifact_dir(output_dir: Path, artifact_dir: str) -> str:
    artifact_path = Path(artifact_dir).resolve()
    try:
        return str(artifact_path.relative_to(output_dir.resolve()))
    except ValueError:
        return str(artifact_path)


def main() -> None:
    output_dir = ROOT / "artifacts" / "foundation_annotation_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = "pbmc3k_processed"
    seed = 42
    adata, spec = _load_dataset(dataset_name)
    rows: list[dict[str, object]] = []
    model_names = tuple(
        model_name
        for model_name in PROFILE_DEFAULTS["ci"]["foundation_annotation"][dataset_name]
        if model_name != "pca_logistic_annotation"
    )

    for model_name in model_names:
        row = run_scgpt_annotation_strategy(
            dataset_name=dataset_name,
            adata=adata,
            label_key=spec.label_key,
            batch_key=spec.batch_key,
            seed=seed,
            output_root=output_dir,
            model_name=model_name,
            profile="ci",
        )
        row["profile"] = "foundation_annotation_smoke"
        row["artifact_dir"] = _relative_artifact_dir(output_dir, str(row["artifact_dir"]))
        rows.append(row)

    metrics_frame = pd.DataFrame.from_records(rows).sort_values(["model", "seed"]).reset_index(
        drop=True
    )
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)

    metrics_by_model = {str(row["model"]): row for row in rows}
    summary = {
        "dataset": dataset_name,
        "label_key": DATASET_SPECS[dataset_name].label_key,
        "checkpoint": "whole-human",
    }
    if "scgpt_frozen_probe" in metrics_by_model:
        summary["probe_accuracy"] = float(metrics_by_model["scgpt_frozen_probe"]["accuracy"])
        summary["probe_macro_f1"] = float(metrics_by_model["scgpt_frozen_probe"]["macro_f1"])
        summary["probe_runtime_sec"] = float(metrics_by_model["scgpt_frozen_probe"]["runtime_sec"])
    if "scgpt_head" in metrics_by_model:
        summary["head_accuracy"] = float(metrics_by_model["scgpt_head"]["accuracy"])
        summary["head_macro_f1"] = float(metrics_by_model["scgpt_head"]["macro_f1"])
        summary["trainable_parameters_head"] = int(
            metrics_by_model["scgpt_head"]["trainable_parameters"]
        )
        summary["head_runtime_sec"] = float(metrics_by_model["scgpt_head"]["runtime_sec"])
    if "scgpt_lora" in metrics_by_model:
        summary["lora_accuracy"] = float(metrics_by_model["scgpt_lora"]["accuracy"])
        summary["lora_macro_f1"] = float(metrics_by_model["scgpt_lora"]["macro_f1"])
        summary["trainable_parameters_lora"] = int(
            metrics_by_model["scgpt_lora"]["trainable_parameters"]
        )
        summary["lora_runtime_sec"] = float(metrics_by_model["scgpt_lora"]["runtime_sec"])

    wrapper_output_dir = output_dir / "wrapper"
    wrapper_adata = _subset_adata_for_foundation(
        adata,
        label_key=spec.label_key,
        seed=seed,
        max_cells=64,
    )
    wrapper_adata = _subset_foundation_genes(wrapper_adata, max_genes=48)
    inspection = inspect_annotation_data(
        wrapper_adata,
        label_key=spec.label_key,
        checkpoint="whole-human",
        min_gene_overlap=min(16, wrapper_adata.n_vars),
    )
    wrapper_runner = adapt_annotation(
        wrapper_adata,
        label_key=spec.label_key,
        batch_size=16,
        device="auto",
        output_dir=wrapper_output_dir,
    )
    if not isinstance(wrapper_runner, ScGPTAnnotationRunner):
        msg = "Top-level annotation alias no longer returns the expected wrapper type."
        raise RuntimeError(msg)
    if wrapper_runner.strategies != ("frozen_probe", "head"):
        msg = "Top-level annotation alias no longer uses the expected default strategy ladder."
        raise RuntimeError(msg)
    wrapper_save_dir = wrapper_runner.save(wrapper_output_dir / "best_model")
    reloaded_runner = AnnotationRunner.load(wrapper_save_dir, device="auto")
    original_predictions = wrapper_runner.predict(wrapper_adata)
    reloaded_predictions = reloaded_runner.predict(wrapper_adata)
    reload_matches = bool(
        np.array_equal(original_predictions["label_codes"], reloaded_predictions["label_codes"])
        and np.allclose(
            original_predictions["probabilities"],
            reloaded_predictions["probabilities"],
            atol=1e-6,
        )
    )
    summary["wrapper_alias_class"] = type(wrapper_runner).__name__
    summary["wrapper_default_strategies"] = list(wrapper_runner.strategies)
    summary["inspection_num_genes_matched"] = int(inspection.num_genes_matched)
    summary["inspection_stratify_possible"] = bool(inspection.stratify_possible)
    summary["wrapper_best_strategy"] = str(wrapper_runner.best_strategy_)
    summary["wrapper_reload_match"] = reload_matches
    summary["wrapper_manifest"] = _relative_artifact_dir(
        output_dir,
        str(wrapper_save_dir / "manifest.json"),
    )
    summary["wrapper_model_state"] = _relative_artifact_dir(
        output_dir,
        str(wrapper_save_dir / "model_state.pt"),
    )
    summary["wrapper_strategy_metrics"] = _relative_artifact_dir(
        output_dir,
        str(wrapper_output_dir / "strategy_metrics.csv"),
    )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# scDLKit foundation annotation smoke summary",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Label key: `{spec.label_key}`",
        "- Checkpoint: `whole-human`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary.items():
        if key in {"dataset", "label_key", "checkpoint"}:
            continue
        if isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.4f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
