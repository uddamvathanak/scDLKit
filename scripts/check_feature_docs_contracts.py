"""Validate tutorial and API-reference coverage for public scDLKit features."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs" / "feature_contracts.yml"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "docs_contracts"
REQUIRED_API_HEADINGS = (
    "## What it is",
    "## When to use it",
    "## Minimal example",
    "## Parameters",
    "## Input expectations",
    "## Returns / outputs",
    "## Failure modes / raises",
    "## Notes / caveats",
    "## Related tutorial(s)",
)


@dataclass(frozen=True, slots=True)
class FeatureContract:
    feature_id: str
    group: str
    public_surface: tuple[str, ...]
    status: str
    tutorial_refs: tuple[str, ...]
    api_page: str
    has_api_contract: bool
    has_parameter_expectations: bool
    has_return_expectations: bool
    has_failure_modes: bool
    page_only_grouping: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_PATH,
        help="Path to the docs feature-contract registry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary artifacts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the docs contract is incomplete.",
    )
    return parser.parse_args()


def _load_tutorial_ids() -> set[str]:
    module_path = ROOT / "scripts" / "execute_tutorial_suite.py"
    spec = importlib.util.spec_from_file_location("execute_tutorial_suite", module_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load tutorial specs from {module_path}."
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {str(tutorial.name) for tutorial in module.TUTORIAL_SPECS}


def _load_registry(path: Path) -> list[FeatureContract]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        msg = "Feature registry must be a JSON-compatible YAML list."
        raise ValueError(msg)
    contracts: list[FeatureContract] = []
    for entry in raw:
        if not isinstance(entry, dict):
            msg = "Each feature registry entry must be a mapping."
            raise ValueError(msg)
        contracts.append(
            FeatureContract(
                feature_id=str(entry["feature_id"]),
                group=str(entry["group"]),
                public_surface=tuple(str(value) for value in entry["public_surface"]),
                status=str(entry["status"]),
                tutorial_refs=tuple(str(value) for value in entry["tutorial_refs"]),
                api_page=str(entry["api_page"]),
                has_api_contract=bool(entry["has_api_contract"]),
                has_parameter_expectations=bool(entry["has_parameter_expectations"]),
                has_return_expectations=bool(entry["has_return_expectations"]),
                has_failure_modes=bool(entry["has_failure_modes"]),
                page_only_grouping=bool(entry.get("page_only_grouping", False)),
            )
        )
    return contracts


def _resolve_public_surface(surface: str) -> Any:
    parts = surface.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            target: Any = importlib.import_module(module_name)
        except ImportError:
            continue
        for attribute in parts[index:]:
            target = getattr(target, attribute)
        return target
    msg = f"Unable to import public surface '{surface}'."
    raise ImportError(msg)


def _page_text(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8")
    return cache[path]


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# scDLKit docs contract summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Registry entries: `{summary['total_features']}`",
        f"- Fully covered entries: `{summary['features_fully_covered']}`",
        f"- Missing tutorial mappings: `{len(summary['missing_tutorial_mappings'])}`",
        f"- Missing API pages: `{len(summary['missing_api_pages'])}`",
        f"- Missing contract sections: `{len(summary['missing_contract_sections'])}`",
        f"- Passed: `{summary['passed']}`",
        "",
        "## Coverage split",
        "",
    ]
    for status, count in summary["status_split"].items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Feature issues", ""])
    if not summary["features"]:
        lines.append("- No registry entries were found.")
    has_feature_issue = False
    for feature in summary["features"]:
        if feature["issues"]:
            has_feature_issue = True
            lines.append(f"- `{feature['feature_id']}`")
            lines.extend(f"  - {issue}" for issue in feature["issues"])
    if summary["features"] and not has_feature_issue:
        lines.append("- No feature-level issues detected.")
    return "\n".join(lines) + "\n"


def build_summary(*, registry_path: Path) -> dict[str, Any]:
    tutorial_ids = _load_tutorial_ids()
    contracts = _load_registry(registry_path)
    page_cache: dict[Path, str] = {}
    page_statuses: dict[Path, set[str]] = {}
    for contract in contracts:
        page_statuses.setdefault(ROOT / contract.api_page, set()).add(contract.status)

    feature_records: list[dict[str, Any]] = []
    missing_tutorials: list[dict[str, str]] = []
    missing_pages: list[dict[str, str]] = []
    missing_sections: list[dict[str, Any]] = []
    import_failures: list[dict[str, str]] = []
    invalid_status_labels: list[dict[str, str]] = []

    for contract in contracts:
        issues: list[str] = []
        if not contract.tutorial_refs:
            issues.append("No tutorial references are registered.")
            missing_tutorials.append(
                {"feature_id": contract.feature_id, "tutorial_ref": "<none>"}
            )
        for tutorial_ref in contract.tutorial_refs:
            if tutorial_ref not in tutorial_ids:
                issues.append(f"Unknown tutorial reference `{tutorial_ref}`.")
                missing_tutorials.append(
                    {"feature_id": contract.feature_id, "tutorial_ref": tutorial_ref}
                )

        page_path = ROOT / contract.api_page
        if not page_path.exists():
            issues.append(f"API page `{contract.api_page}` is missing.")
            missing_pages.append(
                {"feature_id": contract.feature_id, "api_page": contract.api_page}
            )
        else:
            page_text = _page_text(page_path, page_cache)
            absent_headings = [
                heading for heading in REQUIRED_API_HEADINGS if heading not in page_text
            ]
            if absent_headings:
                issues.append(
                    "API page is missing required contract headings: "
                    + ", ".join(f"`{heading}`" for heading in absent_headings)
                    + "."
                )
                missing_sections.append(
                    {
                        "feature_id": contract.feature_id,
                        "api_page": contract.api_page,
                        "sections": absent_headings,
                    }
                )
            for status_label in page_statuses.get(page_path, set()):
                if f"status: {status_label}" not in page_text.lower():
                    issues.append(
                        f"API page does not label the surface as `{status_label}`."
                    )
                    invalid_status_labels.append(
                        {
                            "feature_id": contract.feature_id,
                            "api_page": contract.api_page,
                            "status": status_label,
                        }
                    )
        if not contract.page_only_grouping:
            for surface in contract.public_surface:
                try:
                    _resolve_public_surface(surface)
                except Exception as exc:  # pragma: no cover - exercised in the checker
                    issues.append(f"Public surface `{surface}` is not importable: {exc}")
                    import_failures.append(
                        {
                            "feature_id": contract.feature_id,
                            "public_surface": surface,
                        }
                    )
        feature_records.append(
            {
                "feature_id": contract.feature_id,
                "group": contract.group,
                "status": contract.status,
                "api_page": contract.api_page,
                "tutorial_refs": list(contract.tutorial_refs),
                "issues": issues,
                "passed": not issues,
            }
        )

    status_split: dict[str, int] = {}
    for contract in contracts:
        status_split[contract.status] = status_split.get(contract.status, 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(registry_path.relative_to(ROOT)),
        "total_features": len(contracts),
        "features_fully_covered": sum(1 for record in feature_records if record["passed"]),
        "missing_tutorial_mappings": missing_tutorials,
        "missing_api_pages": missing_pages,
        "missing_contract_sections": missing_sections,
        "import_failures": import_failures,
        "status_label_issues": invalid_status_labels,
        "status_split": status_split,
        "features": feature_records,
        "passed": not any(
            (
                missing_tutorials,
                missing_pages,
                missing_sections,
                import_failures,
                invalid_status_labels,
            )
        ),
    }


def main() -> None:
    args = parse_args()
    summary = build_summary(registry_path=args.registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / "summary.json"
    summary_md = args.output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(render_summary_markdown(summary), encoding="utf-8")
    print(render_summary_markdown(summary))
    if args.check and not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
