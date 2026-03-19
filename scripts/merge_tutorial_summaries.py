"""Merge partial tutorial-suite summaries into a single release-gate summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input tutorial summary JSON files to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Merged tutorial summary JSON path.",
    )
    return parser.parse_args()


def _unique_notebooks(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        for notebook in summary.get("notebooks", []):
            merged[str(notebook["name"])] = dict(notebook)
    return [merged[name] for name in sorted(merged)]


def merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        msg = "At least one tutorial summary is required."
        raise ValueError(msg)
    notebooks = _unique_notebooks(summaries)
    runtime_total = sum(
        float(summary.get("runtime", {}).get("total_sec", 0.0)) for summary in summaries
    )
    runtime_budget = max(
        float(summary.get("runtime", {}).get("budget_sec", 0.0)) for summary in summaries
    )
    missing_files = sorted(
        {
            str(path)
            for summary in summaries
            for path in summary.get("artifact_checks", {}).get("missing_files", [])
        }
    )
    issues = [
        str(issue)
        for summary in summaries
        for issue in summary.get("issues", [])
    ]
    runtime_passed = runtime_total <= runtime_budget
    artifact_passed = not missing_files
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": summaries[0].get("profile", "ci"),
        "group": "all",
        "notebooks": notebooks,
        "runtime": {
            "total_sec": runtime_total,
            "budget_sec": runtime_budget,
            "passed": runtime_passed,
            "notebook_count": len(notebooks),
        },
        "artifact_checks": {
            "passed": artifact_passed,
            "missing_files": missing_files,
        },
        "issues": issues,
        "validated": all(bool(summary.get("validated", False)) for summary in summaries),
        "passed": runtime_passed and artifact_passed,
    }


def main() -> None:
    args = parse_args()
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    merged = merge_summaries(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
