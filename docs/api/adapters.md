# Adapters

Adapters are the supported path for bringing a raw PyTorch module into scDLKit without registering it as a built-in model.

Use them when:

- you already have an `nn.Module`
- you want to train it through `Trainer`
- you want to reuse scDLKit evaluation and Scanpy handoff patterns

```{eval-rst}
.. autoclass:: scdlkit.adapters.TorchModuleAdapter
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.adapters.ReconstructionModuleAdapter
   :members:
```

```{eval-rst}
.. autoclass:: scdlkit.adapters.ClassificationModuleAdapter
   :members:
```

```{eval-rst}
.. autofunction:: scdlkit.adapters.wrap_reconstruction_module
```

```{eval-rst}
.. autofunction:: scdlkit.adapters.wrap_classification_module
```
   :members:
```
