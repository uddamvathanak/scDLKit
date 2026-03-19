"""Execute tutorial notebooks and validate their expected output artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
NOTEBOOK_ARTIFACTS_DIR = ARTIFACTS_DIR / "notebooks"
SUMMARY_DIR = ARTIFACTS_DIR / "tutorial_validation"

RUNTIME_BUDGETS = {
    "ci": 480.0,
    "full": 1200.0,
}


@dataclass(frozen=True, slots=True)
class TutorialSpec:
    name: str
    group: str
    source: Path
    executed_stem: str
    required_artifacts: tuple[Path, ...]


TUTORIAL_SPECS = (
    TutorialSpec(
        name="scanpy_pbmc_quickstart",
        group="classic",
        source=ROOT / "examples" / "train_vae_pbmc.ipynb",
        executed_stem="scanpy_pbmc_quickstart",
        required_artifacts=(
            ROOT / "artifacts" / "pbmc_vae_quickstart" / "report.md",
            ROOT / "artifacts" / "pbmc_vae_quickstart" / "report.csv",
            ROOT / "artifacts" / "pbmc_vae_quickstart" / "loss_curve.png",
            ROOT / "artifacts" / "pbmc_vae_quickstart" / "latent_umap.png",
        ),
    ),
    TutorialSpec(
        name="downstream_scanpy_after_scdlkit",
        group="classic",
        source=ROOT / "examples" / "downstream_scanpy_after_scdlkit.ipynb",
        executed_stem="downstream_scanpy_after_scdlkit",
        required_artifacts=(
            ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "report.md",
            ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "report.csv",
            ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "latent_umap.png",
            ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "leiden_umap.png",
            ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "marker_dotplot.png",
            ROOT / "artifacts" / "downstream_scanpy_after_scdlkit" / "rank_genes_groups.csv",
        ),
    ),
    TutorialSpec(
        name="pbmc_model_comparison",
        group="classic",
        source=ROOT / "examples" / "compare_models_pbmc.ipynb",
        executed_stem="pbmc_model_comparison",
        required_artifacts=(
            ROOT / "artifacts" / "pbmc_compare" / "benchmark_metrics.csv",
            ROOT / "artifacts" / "pbmc_compare" / "benchmark_comparison.png",
            ROOT / "artifacts" / "pbmc_compare" / "pca_reference_umap.png",
            ROOT / "artifacts" / "pbmc_compare" / "best_baseline_umap.png",
        ),
    ),
    TutorialSpec(
        name="reconstruction_sanity_pbmc",
        group="classic",
        source=ROOT / "examples" / "reconstruction_sanity_pbmc.ipynb",
        executed_stem="reconstruction_sanity_pbmc",
        required_artifacts=(
            ROOT / "artifacts" / "reconstruction_sanity_pbmc" / "report.md",
            ROOT / "artifacts" / "reconstruction_sanity_pbmc" / "report.csv",
            ROOT / "artifacts" / "reconstruction_sanity_pbmc" / "loss_curve.png",
            ROOT / "artifacts" / "reconstruction_sanity_pbmc" / "reconstruction_scatter.png",
            ROOT / "artifacts" / "reconstruction_sanity_pbmc" / "gene_panel_reconstruction.png",
        ),
    ),
    TutorialSpec(
        name="pbmc_classification",
        group="classic",
        source=ROOT / "examples" / "classification_demo.ipynb",
        executed_stem="pbmc_classification",
        required_artifacts=(
            ROOT / "artifacts" / "pbmc_classification" / "report.md",
            ROOT / "artifacts" / "pbmc_classification" / "report.csv",
            ROOT / "artifacts" / "pbmc_classification" / "loss_curve.png",
            ROOT / "artifacts" / "pbmc_classification" / "confusion_matrix.png",
        ),
    ),
    TutorialSpec(
        name="custom_model_extension",
        group="classic",
        source=ROOT / "examples" / "custom_model_extension.ipynb",
        executed_stem="custom_model_extension",
        required_artifacts=(
            ROOT / "artifacts" / "custom_model_extension" / "report.md",
            ROOT / "artifacts" / "custom_model_extension" / "report.csv",
            ROOT / "artifacts" / "custom_model_extension" / "loss_curve.png",
            ROOT / "artifacts" / "custom_model_extension" / "latent_umap.png",
        ),
    ),
    TutorialSpec(
        name="scgpt_pbmc_embeddings",
        group="foundation",
        source=ROOT / "examples" / "scgpt_pbmc_embeddings.ipynb",
        executed_stem="scgpt_pbmc_embeddings",
        required_artifacts=(
            ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "report.md",
            ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "report.csv",
            ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "latent_umap.png",
            ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "linear_probe_confusion_matrix.png",
            ROOT / "artifacts" / "scgpt_pbmc_embeddings" / "embedding_summary.json",
        ),
    ),
    TutorialSpec(
        name="scgpt_cell_type_annotation",
        group="foundation",
        source=ROOT / "examples" / "scgpt_cell_type_annotation.ipynb",
        executed_stem="scgpt_cell_type_annotation",
        required_artifacts=(
            ROOT / "artifacts" / "scgpt_cell_type_annotation" / "report.md",
            ROOT / "artifacts" / "scgpt_cell_type_annotation" / "report.csv",
            ROOT / "artifacts" / "scgpt_cell_type_annotation" / "strategy_metrics.csv",
            ROOT / "artifacts" / "scgpt_cell_type_annotation" / "frozen_embedding_umap.png",
            ROOT / "artifacts" / "scgpt_cell_type_annotation" / "lora_embedding_umap.png",
            (
                ROOT
                / "artifacts"
                / "scgpt_cell_type_annotation"
                / "best_strategy_confusion_matrix.png"
            ),
        ),
    ),
    TutorialSpec(
        name="synthetic_smoke",
        group="classic",
        source=ROOT / "examples" / "first_run_synthetic.ipynb",
        executed_stem="first_run_synthetic",
        required_artifacts=(
            ROOT / "artifacts" / "first_run_notebook" / "report.md",
            ROOT / "artifacts" / "first_run_notebook" / "report.csv",
            ROOT / "artifacts" / "first_run_notebook" / "loss_curve.png",
            ROOT / "artifacts" / "first_run_notebook" / "latent_pca.png",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(RUNTIME_BUDGETS),
        default="ci",
        help="Tutorial execution profile.",
    )
    parser.add_argument(
        "--group",
        choices=("all", "classic", "foundation"),
        default="all",
        help="Subset of tutorials to execute.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when tutorial runtime or artifact validation does not pass.",
    )
    return parser.parse_args()


def _selected_specs(group: str) -> tuple[TutorialSpec, ...]:
    if group == "all":
        return TUTORIAL_SPECS
    return tuple(spec for spec in TUTORIAL_SPECS if spec.group == group)


def _execute_notebook(spec: TutorialSpec, profile: str) -> tuple[dict[str, object], list[str]]:
    NOTEBOOK_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(spec.source, as_version=4)
    executor = ExecutePreprocessor(timeout=None, kernel_name="python3")
    print(
        f"[execute_tutorial_suite] starting {spec.name} ({spec.group})",
        flush=True,
    )
    started_at = perf_counter()
    executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    runtime_sec = perf_counter() - started_at
    executed_name = f"{spec.executed_stem}.{profile}.executed.ipynb"
    executed_path = NOTEBOOK_ARTIFACTS_DIR / executed_name
    nbformat.write(notebook, executed_path)

    missing_files = [
        str(path.relative_to(ROOT))
        for path in spec.required_artifacts
        if not path.exists()
    ]
    record: dict[str, object] = {
        "name": spec.name,
        "group": spec.group,
        "source": str(spec.source.relative_to(ROOT)),
        "executed_notebook": str(executed_path.relative_to(ROOT)),
        "runtime_sec": runtime_sec,
        "artifacts": [str(path.relative_to(ROOT)) for path in spec.required_artifacts],
        "missing_artifacts": missing_files,
        "passed": not missing_files,
    }
    print(
        f"[execute_tutorial_suite] finished {spec.name} in {runtime_sec:.1f}s",
        flush=True,
    )
    return record, missing_files


def render_summary_markdown(summary: dict[str, object]) -> str:
    runtime = summary["runtime"]
    lines = [
        "# scDLKit tutorial-suite summary",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Group: `{summary['group']}`",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Tutorial suite passed: `{summary['passed']}`",
        f"- Total runtime: `{runtime['total_sec']:.1f}s` / `{runtime['budget_sec']:.0f}s`",
        "",
        "## Notebook runs",
        "",
    ]
    for notebook in summary["notebooks"]:
        lines.append(
            "- "
            f"`{notebook['name']}`: `{notebook['runtime_sec']:.1f}s`, "
            f"passed `{notebook['passed']}`"
        )
        if notebook["missing_artifacts"]:
            lines.extend(
                f"  - missing artifact: `{path}`" for path in notebook["missing_artifacts"]
            )
    if summary["issues"]:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in summary["issues"])
    return "\n".join(lines) + "\n"


def run_tutorial_suite(profile: str, *, group: str) -> dict[str, object]:
    NOTEBOOK_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    notebook_records: list[dict[str, object]] = []
    missing_files: list[str] = []
    suite_started_at = perf_counter()
    specs = _selected_specs(group)
    for spec in specs:
        record, notebook_missing_files = _execute_notebook(spec, profile)
        notebook_records.append(record)
        missing_files.extend(notebook_missing_files)
    total_runtime_sec = perf_counter() - suite_started_at
    runtime_budget_sec = RUNTIME_BUDGETS[profile]
    runtime_passed = total_runtime_sec <= runtime_budget_sec
    issues: list[str] = []
    if not runtime_passed:
        issues.append(
            "Tutorial suite runtime exceeded "
            f"{runtime_budget_sec:.0f}s (observed {total_runtime_sec:.1f}s)."
        )
    if missing_files:
        issues.extend(f"Missing tutorial artifact `{path}`." for path in missing_files)
    summary: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "group": group,
        "notebooks": notebook_records,
        "runtime": {
            "total_sec": total_runtime_sec,
            "budget_sec": runtime_budget_sec,
            "passed": runtime_passed,
            "notebook_count": len(specs),
        },
        "artifact_checks": {
            "passed": not missing_files,
            "missing_files": missing_files,
        },
        "issues": issues,
        "validated": True,
        "passed": runtime_passed and not missing_files,
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = run_tutorial_suite(args.profile, group=args.group)
    (SUMMARY_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (SUMMARY_DIR / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    print(render_summary_markdown(summary))
    if args.check and not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
