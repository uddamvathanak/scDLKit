# Contributing

## Local setup

```bash
python -m pip install -e .[dev,docs,scanpy]
```

## Development workflow

```bash
ruff check .
mypy src
pytest
```

## Pull requests

- Keep public APIs documented.
- Add or update tests with behavior changes.
- Prefer small, reviewable changes.
- Keep examples reproducible on CPU.
