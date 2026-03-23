"""Validate the publication planning and tutorial-map structure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ROADMAP_HEADINGS = (
    "## Paper vision",
    "## Current implementation truth",
    "## Current objective",
)
REQUIRED_TUTORIAL_HEADINGS = (
    "## Main research tasks",
    "## Supporting workflows",
    "## Advanced / appendix workflows",
)
REQUIRED_TASK_LABELS = (
    "Cell type annotation",
    "Integration / representation transfer",
    "Perturbation-response prediction",
    "Spatial domain / niche classification",
)
REQUIRED_CURRENT_FOCUS_SECTIONS = (
    "Current milestone:",
    "Why now:",
    "Top 3 deliverables:",
    "Blockers:",
    "Exit criteria:",
    "Next milestone after this one:",
)
CHECKLIST_FILES = (
    "planning/checklists/00-publication-operating-system.md",
    "planning/checklists/01-annotation.md",
    "planning/checklists/02-spatial.md",
    "planning/checklists/03-integration.md",
    "planning/checklists/04-perturbation.md",
    "planning/checklists/05-cross-model-peft.md",
    "planning/checklists/06-paper-assets-and-manuscript.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the planning structure is incomplete.",
    )
    return parser.parse_args()


def _missing_headings(path: Path, headings: tuple[str, ...]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [heading for heading in headings if heading not in text]


def _tutorial_targets_exist(path: Path) -> list[str]:
    issues: list[str] = []
    text = path.read_text(encoding="utf-8").splitlines()
    for line in text:
        stripped = line.strip()
        if not stripped.startswith(":link: /_tutorials/"):
            continue
        tutorial_id = stripped.removeprefix(":link: /_tutorials/")
        tutorial_path = ROOT / "docs" / "_tutorials" / f"{tutorial_id}.ipynb"
        if not tutorial_path.exists():
            issues.append(
                f"Tutorial card links to missing notebook `{tutorial_id}`."
            )
    return issues


def main() -> int:
    args = parse_args()
    issues: list[str] = []

    roadmap_path = ROOT / "docs" / "roadmap.md"
    tutorials_path = ROOT / "docs" / "tutorials" / "index.md"
    current_focus_path = ROOT / "planning" / "current-focus.md"

    for required_path in (roadmap_path, tutorials_path, current_focus_path):
        if not required_path.exists():
            issues.append(f"Missing required file `{required_path.relative_to(ROOT)}`.")

    if roadmap_path.exists():
        for heading in _missing_headings(roadmap_path, REQUIRED_ROADMAP_HEADINGS):
            issues.append(f"Roadmap is missing heading `{heading}`.")

    if tutorials_path.exists():
        for heading in _missing_headings(tutorials_path, REQUIRED_TUTORIAL_HEADINGS):
            issues.append(f"Tutorial index is missing heading `{heading}`.")
        text = tutorials_path.read_text(encoding="utf-8")
        for label in REQUIRED_TASK_LABELS:
            if label not in text:
                issues.append(f"Tutorial index is missing main task label `{label}`.")
        issues.extend(_tutorial_targets_exist(tutorials_path))

    if current_focus_path.exists():
        for section in _missing_headings(current_focus_path, REQUIRED_CURRENT_FOCUS_SECTIONS):
            issues.append(f"Current focus file is missing section `{section}`.")

    for relative_path in CHECKLIST_FILES:
        checklist_path = ROOT / relative_path
        if not checklist_path.exists():
            issues.append(f"Missing checklist file `{relative_path}`.")

    if issues:
        print("# scDLKit planning structure check")
        print()
        for issue in issues:
            print(f"- {issue}")
        return 1 if args.check else 0

    print("# scDLKit planning structure check")
    print()
    print("- Passed: `True`")
    print(f"- Roadmap: `{roadmap_path.relative_to(ROOT)}`")
    print(f"- Current focus: `{current_focus_path.relative_to(ROOT)}`")
    print("- Checklist files present: `7`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
