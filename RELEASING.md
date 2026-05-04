# Releasing scDLKit

## Release flow

1. Verify the local gates:
   - `python -m ruff check src tests scripts`
   - `python -m mypy src`
   - `python -m pytest`
   - `python scripts/run_foundation_smoke.py`
   - `python scripts/run_foundation_annotation_smoke.py`
   - `python scripts/prepare_tutorial_notebooks.py --execute published`
   - `python -m sphinx -b html docs docs/_build/html -W --keep-going`
   - `python -m build`
   - `python -m twine check dist/*`
2. Ensure GitHub Pages is enabled with `Source = GitHub Actions`.
3. Ensure the GitHub environments exist:
   - `github-pages`
   - `testpypi`
   - `pypi`
4. Ensure trusted publishing is configured:
   - TestPyPI publisher:
     - repository: `uddamvathanak/scDLKit`
     - workflow: `release-testpypi.yml`
     - environment: `testpypi`
   - PyPI publisher:
     - repository: `uddamvathanak/scDLKit`
     - workflow: `release.yml`
     - environment: `pypi`
5. Trigger the `release-testpypi` workflow with:
   - `ref = main`
   - `version = 0.1.7`
6. Wait for the TestPyPI publish and smoke-install jobs to pass.
7. Create and push the final tag:
   - `git tag v0.1.7`
   - `git push origin v0.1.7`
8. Approve the `pypi` environment in GitHub Actions.
9. Verify the final release:
   - `python -m pip install scdlkit`
   - PyPI project page renders correctly
   - docs site is live
   - README badges resolve
   - GitHub Release contains `dist/` artifacts

## Notes

- Keep `CITATION.cff` version and `date-released` aligned with the final tag date.
- If a local editable install reports stale metadata, refresh it with
  `python -m pip install -e .[dev,docs,foundation,tutorials]`.
- If `scdlkit` is rejected on first claim because the name was taken, stop the release and rename the package across the repo before publishing.
- Do not use long-lived PyPI API tokens unless trusted publishing is unavailable.
