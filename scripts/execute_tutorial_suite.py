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
from tutorial_catalog import EXECUTED_STEMS, NOTEBOOK_GROUPS, NOTEBOOK_MAP, REQUIRED_ARTIFACTS, ROOT

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


TUTORIAL_SPECS = tuple(
    TutorialSpec(
        name=published_name.removesuffix(".ipynb"),
        group=NOTEBOOK_GROUPS[published_name],
        source=source_path,
        executed_stem=EXECUTED_STEMS[published_name],
        required_artifacts=REQUIRED_ARTIFACTS[published_name],
    )
    for published_name, source_path in NOTEBOOK_MAP.items()
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
        "--only",
        nargs="+",
        default=None,
        help="Optional tutorial names to execute explicitly.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when tutorial runtime or artifact validation does not pass.",
    )
    return parser.parse_args()


def _selected_specs(group: str, only: list[str] | None) -> tuple[TutorialSpec, ...]:
    if only:
        allowed_names = set(only)
        selected = tuple(spec for spec in TUTORIAL_SPECS if spec.name in allowed_names)
        missing = sorted(allowed_names - {spec.name for spec in selected})
        if missing:
            msg = (
                "Unknown tutorial names passed to --only: "
                f"{', '.join(missing)}."
            )
            raise ValueError(msg)
        return selected
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


def run_tutorial_suite(
    profile: str,
    *,
    group: str,
    only: list[str] | None,
) -> dict[str, object]:
    NOTEBOOK_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    notebook_records: list[dict[str, object]] = []
    missing_files: list[str] = []
    suite_started_at = perf_counter()
    specs = _selected_specs(group, only)
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
    summary = run_tutorial_suite(args.profile, group=args.group, only=args.only)
    (SUMMARY_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (SUMMARY_DIR / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    print(render_summary_markdown(summary))
    if args.check and not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
