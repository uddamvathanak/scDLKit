"""Run the annotation benchmark path once and reuse its metrics in CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_quality_suite import (  # noqa: E402
    DATASET_SPECS,
    PROFILE_DEFAULTS,
    _load_dataset,
    run_scgpt_annotation_strategy,
)


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
