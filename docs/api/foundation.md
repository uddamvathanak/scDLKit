# Experimental foundation helpers

The foundation helpers are experimental and currently scoped to frozen scGPT embedding extraction.

Use them when you want to:

- prepare PBMC data for the official `whole-human` checkpoint
- load the frozen checkpoint
- extract embeddings and hand them back to Scanpy

Do not treat this section as a stable general foundation-model abstraction yet.

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTPreparedData
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.foundation.ScGPTEmbeddingModel
   :members:
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.ensure_scgpt_checkpoint
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.list_scgpt_checkpoints
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.load_scgpt_model
```

```{eval-rst}
.. autofunction:: scdlkit.foundation.prepare_scgpt_data
```
   :members:
```
