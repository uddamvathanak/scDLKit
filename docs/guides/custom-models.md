# Custom Models

`v0.1.3` adds the first extensibility path for user-supplied PyTorch modules.

The supported surface is `Trainer`, not `TaskRunner`. `TaskRunner` remains the beginner-facing path for built-in scDLKit models. If you want to bring your own `nn.Module`, wrap it with an adapter and train it through the lower-level workflow.

## What adapters do

The adapter layer lets you:

- keep your model as a normal `torch.nn.Module`
- reuse scDLKit data preparation and training loops
- evaluate predictions with the same scDLKit metrics helpers
- write the resulting latent space back into `adata.obsm` and continue with Scanpy

This keeps scDLKit focused on rapid prototyping and validation without forcing every custom model into the built-in registry.

## Current limitations

The first adapter release is intentionally narrow:

- wrapped models are supported through `Trainer` first
- the module contract is `x`-only input for now
- full-batch callback plumbing is not part of this release
- foundation-model integrations are still future work

That scope is deliberate. The first goal is to make custom small-model prototyping stable before adding larger foundation-model workflows.

## Minimal reconstruction wrapper

```python
import torch
from torch import nn

from scdlkit import Trainer, prepare_data
from scdlkit.adapters import wrap_reconstruction_module


class SmallAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


prepared = prepare_data(adata, label_key="louvain")
wrapped = wrap_reconstruction_module(
    SmallAutoencoder(prepared.input_dim),
    input_dim=prepared.input_dim,
    supported_tasks=("representation", "reconstruction"),
)

trainer = Trainer(
    model=wrapped,
    task="representation",
    device="auto",
    epochs=10,
    batch_size=128,
)
trainer.fit(prepared.train, prepared.val)
```

## Evaluation and Scanpy handoff

```python
from scdlkit.data import transform_adata
from scdlkit.evaluation import evaluate_predictions, save_markdown_report, save_metrics_table

test_predictions = trainer.predict_dataset(prepared.test or prepared.val)
metrics = evaluate_predictions("representation", test_predictions)
save_markdown_report(
    metrics,
    path="artifacts/custom_model_extension/report.md",
    title="scDLKit custom model extension report",
)
save_metrics_table(metrics, "artifacts/custom_model_extension/report.csv")

full_split = transform_adata(
    adata,
    prepared.preprocessing,
    label_encoder=prepared.label_encoder,
    batch_encoder=prepared.batch_encoder,
)
full_predictions = trainer.predict_dataset(full_split)
adata.obsm["X_scdlkit_custom"] = full_predictions["latent"]
```

From there, continue with standard Scanpy steps:

```python
import scanpy as sc

sc.pp.neighbors(adata, use_rep="X_scdlkit_custom")
sc.tl.umap(adata, random_state=42)
sc.pl.umap(adata, color="louvain")
```

## Tutorial

For the full end-to-end walkthrough, see the rendered notebook:

- [Custom model extension](/_tutorials/custom_model_extension)

This notebook defines a small custom autoencoder directly in the tutorial, trains it through `Trainer`, evaluates the result, and saves artifacts under `artifacts/custom_model_extension/`.
