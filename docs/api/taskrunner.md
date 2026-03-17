# TaskRunner

Use `TaskRunner` when you want the main public workflow:

1. prepare an `AnnData`
2. train a built-in model
3. evaluate it
4. get embeddings or reconstructed expression back

Minimal example:

```python
import scanpy as sc
from scdlkit import TaskRunner

adata = sc.datasets.pbmc3k_processed()
runner = TaskRunner(model="vae", task="representation", label_key="louvain", device="auto")
runner.fit(adata)
latent = runner.encode(adata)
```

For reconstruction-capable models, use:

```python
reconstructed = runner.reconstruct(adata)
```

```{eval-rst}
.. autoclass:: scdlkit.runner.TaskRunner
   :members:
```
