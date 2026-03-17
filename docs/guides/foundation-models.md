# Foundation Models

`scDLKit` now includes an experimental scGPT embedding path for human scRNA-seq workflows.

This first foundation-model release is intentionally narrow:

- embeddings only
- official `whole-human` checkpoint only
- human single-cell RNA only
- `Trainer.predict_dataset(...)` only
- no fine-tuning yet
- no `TaskRunner` support yet

## Install

```bash
python -m pip install "scdlkit[foundation,tutorials]"
```

## What this path is for

Use the experimental foundation path when you want to:

- extract frozen cell embeddings from an official scGPT checkpoint
- compare those embeddings against `PCA` and scDLKit baselines
- keep the same Scanpy downstream handoff through `adata.obsm`

This is the first bridge between the baseline toolkit and later foundation-model adaptation work.

Current validation emphasis:

- release-gating focuses on embedding structure, not on claiming that a frozen linear probe already beats classical baselines
- frozen probe metrics are still recorded and reported so the foundation path is evaluated honestly

## Public API

```python
from scdlkit import Trainer
from scdlkit.foundation import load_scgpt_model, prepare_scgpt_data

prepared = prepare_scgpt_data(
    adata,
    checkpoint="whole-human",
    label_key="louvain",
    batch_size=64,
)

model = load_scgpt_model("whole-human", device="auto")
trainer = Trainer(
    model=model,
    task="representation",
    batch_size=prepared.batch_size,
    device="auto",
)

predictions = trainer.predict_dataset(prepared.dataset)
adata.obsm["X_scgpt_whole_human"] = predictions["latent"]
```

## Current limitations

- input preparation is a separate tokenized pipeline, not `prepare_data(...)`
- the supported surface is frozen inference, not `Trainer.fit(...)`
- this release does not include LoRA, PEFT, or checkpoint fine-tuning
- this release does not include non-human support

## Next step

Once the frozen embedding path is stable, the next extension is fine-tuning and adaptation rather than more checkpoint variants.
