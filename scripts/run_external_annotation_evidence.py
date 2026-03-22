"""Generate beyond-PBMC annotation evidence on human pancreas."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_quality_suite import (  # noqa: E402
    _load_dataset,
    run_foundation_annotation_pca_logistic,
    run_scgpt_annotation_strategy,
)

from scdlkit import AnnotationRunner  # noqa: E402

_DATASET_ORDER = ("pbmc68k_reduced", "openproblems_human_pancreas")
_MODEL_ORDER = (
    "pca_logistic_annotation",
    "scgpt_frozen_probe",
    "scgpt_head",
    "scgpt_lora",
)
_DISPLAY_NAMES = {
    "pca_logistic_annotation": "PCA + logistic regression",
    "scgpt_frozen_probe": "Frozen scGPT probe",
    "scgpt_head": "scGPT head-only tuning",
    "scgpt_lora": "scGPT LoRA tuning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "external_annotation_evidence",
        help="Output directory for evidence artifacts.",
    )
    return parser.parse_args()


def _strategy_sort_value(model_name: str) -> int:
    try:
        return _MODEL_ORDER.index(model_name)
    except ValueError:
        return len(_MODEL_ORDER)


def _strategy_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows)
    frame["strategy"] = frame["model"].map(_DISPLAY_NAMES).fillna(frame["model"])
    ordered = frame.sort_values(
        ["dataset", "model"],
        key=lambda column: column.map(_strategy_sort_value) if column.name == "model" else column,
        kind="mergesort",
    ).reset_index(drop=True)
    columns = [
        "dataset",
        "model",
        "strategy",
        "accuracy",
        "macro_f1",
        "runtime_sec",
        "trainable_parameters",
        "batch_accuracy_mean",
        "batch_accuracy_min",
        "batch_macro_f1_mean",
        "batch_macro_f1_min",
        "artifact_dir",
        "batch_metrics_artifact",
        "confusion_matrix_artifact",
        "latent_umap_artifact",
    ]
    return ordered[[column for column in columns if column in ordered.columns]]


def _combine_batch_metrics(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    combined_rows: list[pd.DataFrame] = []
    for record in frame.to_dict(orient="records"):
        batch_path = Path(str(record["batch_metrics_artifact"]))
        if not batch_path.exists():
            continue
        batch_frame = pd.read_csv(batch_path)
        if batch_frame.empty:
            continue
        batch_frame.insert(0, "strategy", str(record["strategy"]))
        combined_rows.append(batch_frame)
    combined = (
        pd.concat(combined_rows, ignore_index=True)
        if combined_rows
        else pd.DataFrame(columns=["strategy", "batch", "n_cells", "accuracy", "macro_f1"])
    )
    combined.to_csv(output_path, index=False)
    return combined


def _copy_file(source: str | Path, destination: Path) -> None:
    source_path = Path(source)
    if not source_path.exists():
        msg = f"Expected artifact '{source_path}' is missing."
        raise FileNotFoundError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)


def _write_markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join('---' for _ in headers)} |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append(f"| {' | '.join(values)} |")
    return "\n".join(lines)


def _write_pancreas_report(
    *,
    strategy_frame: pd.DataFrame,
    batch_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    best_overall = strategy_frame.sort_values(
        ["macro_f1", "accuracy", "trainable_parameters", "runtime_sec"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    tuned_rows = strategy_frame[strategy_frame["model"].isin(("scgpt_head", "scgpt_lora"))]
    best_trainable = tuned_rows.sort_values(
        ["macro_f1", "accuracy", "trainable_parameters", "runtime_sec"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    report = {
        "dataset": "openproblems_human_pancreas",
        "best_strategy": str(best_overall["strategy"]),
        "best_macro_f1": float(best_overall["macro_f1"]),
        "best_accuracy": float(best_overall["accuracy"]),
        "best_trainable_strategy": str(best_trainable["strategy"]),
        "best_trainable_macro_f1": float(best_trainable["macro_f1"]),
        "best_trainable_accuracy": float(best_trainable["accuracy"]),
    }
    pd.DataFrame([report]).to_csv(output_dir / "report.csv", index=False)
    lines = [
        "# scDLKit external annotation evidence report",
        "",
        "- Dataset: `openproblems_human_pancreas`",
        f"- Best overall strategy: `{report['best_strategy']}`",
        f"- Best overall macro F1: `{report['best_macro_f1']:.4f}`",
        f"- Best trainable strategy: `{report['best_trainable_strategy']}`",
        "",
        "## Strategy comparison",
        "",
        _write_markdown_table(
            strategy_frame[
                [
                    "strategy",
                    "accuracy",
                    "macro_f1",
                    "runtime_sec",
                    "trainable_parameters",
                    "batch_accuracy_mean",
                    "batch_macro_f1_mean",
                ]
            ]
        ),
    ]
    if not batch_metrics.empty:
        lines.extend(
            [
                "",
                "## Per-batch metrics",
                "",
                _write_markdown_table(batch_metrics),
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_dataset_rows(
    *,
    dataset_name: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    adata, spec = _load_dataset(dataset_name, profile="full")
    rows = [
        run_foundation_annotation_pca_logistic(
            dataset_name=dataset_name,
            adata=adata,
            label_key=spec.label_key,
            batch_key=spec.batch_key,
            seed=42,
            output_root=output_root,
            profile="full",
        )
    ]
    for model_name in ("scgpt_frozen_probe", "scgpt_head", "scgpt_lora"):
        rows.append(
            run_scgpt_annotation_strategy(
                dataset_name=dataset_name,
                adata=adata,
                label_key=spec.label_key,
                batch_key=spec.batch_key,
                seed=42,
                output_root=output_root,
                model_name=model_name,
                profile="full",
            )
        )
    return rows


def run_external_annotation_evidence(*, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = output_dir / "_strategy_runs"
    pancreas_root = output_dir / "pancreas"
    cross_dataset_root = output_dir / "cross_dataset"
    pancreas_root.mkdir(parents=True, exist_ok=True)
    cross_dataset_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for dataset_name in _DATASET_ORDER:
        all_rows.extend(
            _collect_dataset_rows(
                dataset_name=dataset_name,
                output_root=benchmark_root,
            )
        )
    strategy_frame = _strategy_frame(all_rows)

    pancreas_frame = strategy_frame[
        strategy_frame["dataset"] == "openproblems_human_pancreas"
    ].copy()
    pancreas_frame.to_csv(pancreas_root / "strategy_metrics.csv", index=False)
    batch_metrics = _combine_batch_metrics(pancreas_frame, pancreas_root / "batch_metrics.csv")
    _write_pancreas_report(
        strategy_frame=pancreas_frame,
        batch_metrics=batch_metrics,
        output_dir=pancreas_root,
    )

    best_overall = pancreas_frame.sort_values(
        ["macro_f1", "accuracy", "trainable_parameters", "runtime_sec"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    frozen_row = pancreas_frame[pancreas_frame["model"] == "scgpt_frozen_probe"].iloc[0]
    _copy_file(
        frozen_row["latent_umap_artifact"],
        pancreas_root / "frozen_embedding_umap.png",
    )
    _copy_file(
        best_overall["latent_umap_artifact"],
        pancreas_root / "best_strategy_embedding_umap.png",
    )
    _copy_file(
        best_overall["confusion_matrix_artifact"],
        pancreas_root / "best_strategy_confusion_matrix.png",
    )

    pancreas_adata, _ = _load_dataset("openproblems_human_pancreas", profile="full")
    trainable_runner = AnnotationRunner(
        label_key="cell_type",
        strategies=("head", "lora"),
        batch_size=64,
        device="auto",
    )
    trainable_runner.fit_compare(pancreas_adata)
    trainable_runner.save(pancreas_root / "best_model")

    cross_dataset_frame = strategy_frame[
        ["dataset", "strategy", "accuracy", "macro_f1", "runtime_sec", "trainable_parameters"]
    ].copy()
    cross_dataset_frame.to_csv(cross_dataset_root / "strategy_summary.csv", index=False)
    lines = [
        "# Cross-dataset annotation strategy summary",
        "",
        _write_markdown_table(cross_dataset_frame),
    ]
    (cross_dataset_root / "strategy_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "datasets": list(_DATASET_ORDER),
        "generated_files": {
            "pancreas": str(pancreas_root),
            "cross_dataset": str(cross_dataset_root),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "pancreas_dir": pancreas_root,
        "cross_dataset_dir": cross_dataset_root,
    }


def main() -> None:
    args = parse_args()
    outputs = run_external_annotation_evidence(output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
