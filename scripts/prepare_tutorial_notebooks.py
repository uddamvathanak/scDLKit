"""Sync and optionally execute tutorial notebooks for the docs site."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from tutorial_catalog import (
    ASSET_MAP,
    NOTEBOOK_GROUPS,
    NOTEBOOK_MAP,
    REQUIRED_ARTIFACTS,
)
from tutorial_publication import (
    attach_publication_metadata,
    infer_last_run_utc,
    infer_latest_path_mtime_utc,
    utc_now_iso,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
TUTORIAL_DIR = DOCS_DIR / "_tutorials"
STATIC_DIR = DOCS_DIR / "_static"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        choices=("none", "published"),
        default="none",
        help="Execute synced notebooks after copying them.",
    )
    parser.add_argument(
        "--group",
        choices=("all", "classic", "foundation"),
        default="all",
        help="Subset of tutorial notebooks to sync.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional target notebook filenames to sync explicitly.",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Skip syncing static docs assets.",
    )
    return parser.parse_args()


def _selected_notebooks(group: str, only: list[str] | None) -> list[tuple[str, Path]]:
    notebooks = list(NOTEBOOK_MAP.items())
    if only:
        allowed = set(only)
        selected = [
            (target_name, source_path)
            for target_name, source_path in notebooks
            if target_name in allowed
        ]
        missing = sorted(allowed - {target_name for target_name, _ in selected})
        if missing:
            msg = (
                "Unknown tutorial notebook targets passed to --only: "
                f"{', '.join(missing)}."
            )
            raise ValueError(msg)
        return selected
    if group == "all":
        return notebooks
    return [
        (target_name, source_path)
        for target_name, source_path in notebooks
        if NOTEBOOK_GROUPS[target_name] == group
    ]


def _missing_artifacts(target_name: str) -> list[Path]:
    return [path for path in REQUIRED_ARTIFACTS[target_name] if not path.exists()]


def sync_notebooks(*, execute: bool, group: str, only: list[str] | None) -> None:
    TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    for target_name, source_path in _selected_notebooks(group, only):
        if not source_path.exists():
            msg = f"Required tutorial notebook is missing: {source_path}"
            raise FileNotFoundError(msg)
        print(
            f"[prepare_tutorial_notebooks] syncing {source_path.name} -> {target_name}",
            flush=True,
        )
        notebook = nbformat.read(source_path, as_version=4)
        target_path = TUTORIAL_DIR / target_name
        last_run_utc = infer_last_run_utc(notebook)
        if last_run_utc is None:
            last_run_utc = infer_latest_path_mtime_utc(REQUIRED_ARTIFACTS[target_name])
        artifact_validation = "not-checked"
        if execute:
            print(
                f"[prepare_tutorial_notebooks] executing {target_name} for published docs",
                flush=True,
            )
            executor = ExecutePreprocessor(timeout=None, kernel_name="python3")
            executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
            last_run_utc = utc_now_iso()
            missing_artifacts = _missing_artifacts(target_name)
            if missing_artifacts:
                joined = ", ".join(str(path.relative_to(ROOT)) for path in missing_artifacts)
                msg = (
                    f"Published tutorial `{target_name}` finished executing but did not "
                    f"produce required artifacts: {joined}"
                )
                raise RuntimeError(msg)
            artifact_validation = "passed"
            print(
                f"[prepare_tutorial_notebooks] finished executing {target_name}",
                flush=True,
            )
        attach_publication_metadata(
            notebook,
            source_path=source_path.relative_to(ROOT),
            target_path=target_path.relative_to(ROOT),
            execution_profile="published" if execute else None,
            artifact_validation=artifact_validation,
            last_run_utc=last_run_utc,
        )
        nbformat.write(notebook, target_path)
        print(f"[prepare_tutorial_notebooks] wrote {target_path}", flush=True)


def sync_assets() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    for target_name, source_path in ASSET_MAP.items():
        if not source_path.exists():
            msg = f"Required docs asset is missing: {source_path}"
            raise FileNotFoundError(msg)
        shutil.copy2(source_path, STATIC_DIR / target_name)


def main() -> None:
    args = parse_args()
    sync_notebooks(
        execute=args.execute == "published",
        group=args.group,
        only=args.only,
    )
    if not args.skip_assets:
        sync_assets()


if __name__ == "__main__":
    main()
