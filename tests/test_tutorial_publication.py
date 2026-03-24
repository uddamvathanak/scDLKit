from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import nbformat

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

render_tutorial_status = importlib.import_module("render_tutorial_status")
tutorial_catalog = importlib.import_module("tutorial_catalog")
tutorial_publication = importlib.import_module("tutorial_publication")


def _make_notebook(title: str) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(f"# {title}"),
        nbformat.v4.new_code_cell("print('ok')"),
    ]
    return notebook


def test_infer_last_run_utc_uses_latest_execution_timestamp() -> None:
    notebook = _make_notebook("Example")
    notebook.cells[1].metadata["execution"] = {
        "iopub.execute_input": "2026-03-23T01:00:00Z",
        "iopub.status.idle": "2026-03-23T01:01:00Z",
    }
    notebook.cells.append(nbformat.v4.new_code_cell("print('again')"))
    notebook.cells[2].metadata["execution"] = {
        "shell.execute_reply": "2026-03-23T02:15:30Z",
    }

    assert tutorial_publication.infer_last_run_utc(notebook) == "2026-03-23T02:15:30Z"


def test_attach_publication_metadata_replaces_existing_status_cell() -> None:
    notebook = _make_notebook("Example")
    tutorial_publication.attach_publication_metadata(
        notebook,
        source_path=Path("examples") / "example.ipynb",
        target_path=Path("docs/_tutorials") / "example.ipynb",
        execution_profile="published",
        artifact_validation="passed",
        last_run_utc="2026-03-23T03:00:00Z",
    )
    tutorial_publication.attach_publication_metadata(
        notebook,
        source_path=Path("examples") / "example.ipynb",
        target_path=Path("docs/_tutorials") / "example.ipynb",
        execution_profile="published",
        artifact_validation="passed",
        last_run_utc="2026-03-23T03:00:00Z",
    )

    status_cells = [
        cell
        for cell in notebook.cells
        if tutorial_publication.STATUS_CELL_TAG in cell.metadata.get("tags", [])
    ]
    assert len(status_cells) == 1
    assert "Last run date (UTC)" in status_cells[0].source
    publication_metadata = notebook.metadata[tutorial_publication.PUBLICATION_METADATA_KEY]
    assert publication_metadata["artifact_validation"] == (
        "passed"
    )


def test_infer_latest_path_mtime_utc_uses_newest_artifact(tmp_path: Path) -> None:
    older = tmp_path / "older.txt"
    newer = tmp_path / "newer.txt"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    os.utime(older, (1_710_000_000, 1_710_000_000))
    os.utime(newer, (1_720_000_000, 1_720_000_000))

    assert tutorial_publication.infer_latest_path_mtime_utc((older, newer)) == (
        "2024-07-03T09:46:40Z"
    )


def test_render_tutorial_status_reports_published_notebooks(tmp_path: Path) -> None:
    tutorial_dir = tmp_path / "_tutorials"
    tutorial_dir.mkdir(parents=True, exist_ok=True)

    for notebook_name, source_path in tutorial_catalog.NOTEBOOK_MAP.items():
        notebook = _make_notebook(tutorial_catalog.NOTEBOOK_TITLES[notebook_name])
        tutorial_publication.attach_publication_metadata(
            notebook,
            source_path=source_path.relative_to(tutorial_catalog.ROOT),
            target_path=Path("docs/_tutorials") / notebook_name,
            execution_profile="published",
            artifact_validation="passed",
            last_run_utc="2026-03-23T04:05:06Z",
        )
        nbformat.write(notebook, tutorial_dir / notebook_name)

    records, issues = render_tutorial_status._load_records(tutorial_dir)
    markdown = render_tutorial_status.render_status_markdown(records, issues)

    assert issues == []
    assert "Tutorial execution status" in markdown
    assert "2026-03-23 04:05 UTC" in markdown
    assert "Main research task: annotation" in markdown
    assert "Supporting workflows" in markdown
    assert "Artifact check" in markdown
