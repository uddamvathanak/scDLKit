"""Helpers for publishing static tutorial notebooks with visible status metadata."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbformat import NotebookNode

PUBLICATION_METADATA_KEY = "scdlkit_tutorial"
STATUS_CELL_TAG = "scdlkit-tutorial-status"


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for notebook metadata."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def infer_last_run_utc(notebook: NotebookNode) -> str | None:
    """Infer the latest code-cell execution timestamp from a notebook."""

    latest: datetime | None = None
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        execution = cell.get("metadata", {}).get("execution", {})
        for key in (
            "shell.execute_reply",
            "iopub.status.idle",
            "iopub.execute_input",
        ):
            candidate = _parse_timestamp(execution.get(key))
            if candidate is not None and (latest is None or candidate > latest):
                latest = candidate
    if latest is None:
        existing = notebook.get("metadata", {}).get(PUBLICATION_METADATA_KEY, {})
        candidate = _parse_timestamp(existing.get("last_run_utc"))
        if candidate is not None:
            latest = candidate
    if latest is None:
        return None
    return latest.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def format_timestamp_for_display(value: str | None) -> str:
    """Format a stored UTC timestamp for tutorial callouts and status pages."""

    parsed = _parse_timestamp(value)
    if parsed is None:
        return "not recorded"
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def infer_latest_path_mtime_utc(paths: Iterable[Path]) -> str | None:
    """Infer a UTC timestamp from the newest existing file in a path list."""

    latest: datetime | None = None
    for path in paths:
        if not path.exists():
            continue
        candidate = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if latest is None or candidate > latest:
            latest = candidate
    if latest is None:
        return None
    return latest.isoformat(timespec="seconds").replace("+00:00", "Z")


def _status_cell_source(metadata: dict[str, object]) -> str:
    profile = metadata.get("execution_profile")
    profile_line = f"- Execution profile: `{profile}`\n" if profile else ""
    return (
        "```{admonition} Published tutorial status\n"
        ":class: note\n\n"
        "This page is a static notebook copy published for documentation review.\n"
        "It is meant to show the exact workflow and outputs from the last recorded run.\n\n"
        f"- Last run date (UTC): `{format_timestamp_for_display(metadata.get('last_run_utc'))}`\n"
        f"- Publication mode: `{metadata['publication_mode']}`\n"
        f"{profile_line}"
        f"- Artifact check in this sync: `{metadata['artifact_validation']}`\n"
        f"- Source notebook: `{metadata['source_notebook']}`\n"
        "```\n"
    )


def _ensure_notebook_language_metadata(notebook: NotebookNode) -> None:
    notebook.metadata.setdefault(
        "kernelspec",
        {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
    )
    notebook.metadata.setdefault(
        "language_info",
        {
            "name": "python",
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "pygments_lexer": "ipython3",
        },
    )


def attach_publication_metadata(
    notebook: NotebookNode,
    *,
    source_path: Path,
    target_path: Path,
    execution_profile: str | None,
    artifact_validation: str,
    last_run_utc: str | None = None,
) -> dict[str, object]:
    """Stamp notebook metadata and insert a visible status callout cell."""

    _ensure_notebook_language_metadata(notebook)
    publication_mode = (
        "static executed tutorial" if execution_profile is not None else "static notebook copy"
    )
    metadata = {
        "source_notebook": source_path.as_posix(),
        "published_notebook": target_path.as_posix(),
        "publication_mode": publication_mode,
        "execution_profile": execution_profile,
        "last_run_utc": last_run_utc or infer_last_run_utc(notebook),
        "published_at_utc": utc_now_iso(),
        "artifact_validation": artifact_validation,
    }
    notebook.metadata[PUBLICATION_METADATA_KEY] = metadata

    status_cell = nbformat.v4.new_markdown_cell(_status_cell_source(metadata))
    status_cell.metadata["tags"] = [STATUS_CELL_TAG]

    retained_cells = []
    for cell in notebook.cells:
        tags = cell.get("metadata", {}).get("tags", [])
        if STATUS_CELL_TAG in tags:
            continue
        retained_cells.append(cell)
    notebook.cells = retained_cells

    insert_at = 1 if notebook.cells and notebook.cells[0].cell_type == "markdown" else 0
    notebook.cells.insert(insert_at, status_cell)
    return metadata


def load_publication_metadata(notebook_path: Path) -> dict[str, object]:
    """Read stored publication metadata from a published tutorial notebook."""

    notebook = nbformat.read(notebook_path, as_version=4)
    stored_metadata = notebook.metadata.get(PUBLICATION_METADATA_KEY, {})
    metadata = dict(stored_metadata)
    metadata["metadata_present"] = bool(stored_metadata)
    metadata.setdefault("last_run_utc", infer_last_run_utc(notebook))
    metadata.setdefault("published_notebook", notebook_path.as_posix())
    return metadata
