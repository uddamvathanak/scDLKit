"""Render a docs page that shows published tutorial execution status."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tutorial_catalog import NOTEBOOK_MAP, NOTEBOOK_SECTIONS, NOTEBOOK_TITLES, ROOT
from tutorial_publication import format_timestamp_for_display, load_publication_metadata

DEFAULT_SOURCE_DIR = ROOT / "docs" / "_tutorials"
DEFAULT_OUTPUT_PATH = ROOT / "docs" / "tutorials" / "status.md"

SECTION_ORDER = (
    "Main research task: annotation",
    "Supporting workflows",
    "Advanced / appendix workflows",
    "Experimental detail appendix",
    "Maintainer / smoke only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing published tutorial notebooks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Markdown file to write.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any published tutorial notebook is missing.",
    )
    return parser.parse_args()


def _load_records(source_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    records: list[dict[str, str]] = []
    issues: list[str] = []
    for notebook_name in NOTEBOOK_MAP:
        notebook_path = source_dir / notebook_name
        if not notebook_path.exists():
            issues.append(f"Missing published tutorial notebook `docs/_tutorials/{notebook_name}`.")
            continue
        metadata = load_publication_metadata(notebook_path)
        if not metadata.get("metadata_present", False):
            issues.append(
                "Published tutorial notebook "
                f"`docs/_tutorials/{notebook_name}` is missing scDLKit publication metadata."
            )
        tutorial_id = notebook_name.removesuffix(".ipynb")
        records.append(
            {
                "tutorial_id": tutorial_id,
                "title": NOTEBOOK_TITLES[notebook_name],
                "section": NOTEBOOK_SECTIONS[notebook_name],
                "docs_path": f"/_tutorials/{tutorial_id}",
                "source_notebook": str(metadata.get("source_notebook", "not recorded")),
                "publication_mode": str(
                    metadata.get("publication_mode", "static notebook copy")
                ),
                "execution_profile": str(metadata.get("execution_profile") or "not recorded"),
                "last_run_utc": format_timestamp_for_display(
                    metadata.get("last_run_utc") if metadata else None
                ),
                "artifact_validation": str(
                    metadata.get("artifact_validation", "not recorded")
                ),
                "published_at_utc": format_timestamp_for_display(
                    metadata.get("published_at_utc") if metadata else None
                ),
            }
        )
    return records, issues


def render_status_markdown(records: list[dict[str, str]], issues: list[str]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    section_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        section_map[record["section"]].append(record)

    lines = [
        "# Tutorial execution status",
        "",
        "This page shows the current static tutorial copies that are published in the docs.",
        (
            "Each tutorial page should expose the same metadata in a visible "
            "status block near the top."
        ),
        "",
        f"- Generated at: `{generated_at}`",
        f"- Published tutorials tracked: `{len(records)}`",
        "",
        "## What this means",
        "",
        (
            "- `Last run date (UTC)` is the latest recorded execution time "
            "embedded in the published notebook."
        ),
        "- `Artifact check` is the result recorded when the notebook was prepared for publication.",
        "- These pages are static documentation copies, not interactive runtime sessions.",
        "",
    ]

    if issues:
        lines.extend(["## Issues", ""])
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("")

    for section in SECTION_ORDER:
        section_records = section_map.get(section)
        if not section_records:
            continue
        lines.extend([f"## {section}", ""])
        lines.append(
            "| Tutorial | Last run date (UTC) | Publication mode | Artifact check | Docs page |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        for record in section_records:
            lines.append(
                "| "
                f"{record['title']} | "
                f"{record['last_run_utc']} | "
                f"{record['publication_mode']} | "
                f"{record['artifact_validation']} | "
                f"[open]({record['docs_path']}) |"
            )
        lines.append("")
        for record in section_records:
            lines.append(
                "- "
                f"`{record['title']}` source: `{record['source_notebook']}`; "
                f"published at `{record['published_at_utc']}`; "
                f"profile `{record['execution_profile']}`"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    records, issues = _load_records(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_status_markdown(records, issues), encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"))
    if args.check and issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
