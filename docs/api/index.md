# API reference

The public API stays intentionally compact, but it should still be easy to navigate.

Use this page as a routing guide:

- start with `TaskRunner` if you want the main beginner workflow
- use `Trainer` and `prepare_data` when you need lower-level control
- use adapters when you want to wrap your own PyTorch module
- treat foundation helpers as experimental

## Start here

### High-level stable workflow

- [TaskRunner](./taskrunner.md)

### Low-level stable workflow

- [Trainer](./trainer.md)
- [Data preparation](./data.md)

### Extension surfaces

- [Adapters](./adapters.md)
- [Experimental foundation helpers](./foundation.md)

### Evaluation and reports

- [Evaluation and outputs](./evaluation.md)

### Built-in models

- [Built-in models](./models.md)

## Minimal quickstart

```python
import scanpy as sc
from scdlkit import TaskRunner

adata = sc.datasets.pbmc3k_processed()

runner = TaskRunner(
    model="vae",
    task="representation",
    label_key="louvain",
    device="auto",
    epochs=20,
    batch_size=128,
    model_kwargs={"kl_weight": 1e-3},
)

runner.fit(adata)
adata.obsm["X_scdlkit_vae"] = runner.encode(adata)
```

## Stable vs experimental

- Stable beginner path:
  - `TaskRunner`
  - `Trainer`
  - `prepare_data`
  - `compare_models`
  - adapter helpers
- Experimental path:
  - frozen scGPT embedding helpers under `scdlkit.foundation`

```{toctree}
:hidden:
:maxdepth: 1

taskrunner
trainer
data
adapters
foundation
evaluation
models
```
