"""Rebuild human-readable quality-suite summaries from a metrics CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=Path, help="Path to a quality-suite metrics.csv file.")
    parser.add_argument(
        "--profile",
        default="full",
        help="Profile name to record in the regenerated summary files.",
    )
    parser.add_argument(
        "--tutorial-summary",
        type=Path,
        default=None,
        help="Optional tutorial validation summary JSON to include in the regenerated summary.",
    )
    return parser.parse_args()


def main() -> None:
    from run_quality_suite import build_summary, load_tutorial_summary, render_summary_markdown

    args = parse_args()
    metrics_frame = pd.read_csv(args.metrics_csv)
    summary = build_summary(
        metrics_frame,
        profile=args.profile,
        tutorial_summary=load_tutorial_summary(args.tutorial_summary),
    )
    output_dir = args.metrics_csv.parent
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    print(render_summary_markdown(summary))


if __name__ == "__main__":
    main()
