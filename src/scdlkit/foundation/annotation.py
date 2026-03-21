"""Experimental scGPT annotation fine-tuning helpers."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor, nn
from torch.nn import functional

from scdlkit.foundation.cache import DEFAULT_SCGPT_CHECKPOINT
from scdlkit.foundation.lora import ScGPTLoRAConfig, apply_scgpt_lora
from scdlkit.foundation.scgpt import ScGPTBackbone, _load_scgpt_backbone
from scdlkit.utils import resolve_device


class ScGPTAnnotationModel(nn.Module):
    """scGPT-based classifier for experimental cell-type annotation tuning.

    Parameters
    ----------
    backbone
        Loaded scGPT backbone used for token embedding and pooling.
    checkpoint_id
        Source checkpoint identifier.
    tuning_strategy
        Either ``"head"`` or ``"lora"``.
    num_classes
        Number of annotation classes.
    label_categories
        Optional class-name order used for reporting and confusion matrices.
    classifier_dropout
        Dropout applied before the final classification layer.

    Notes
    -----
    ``predict_batch(...)`` returns both ``logits`` and normalized ``latent``
    embeddings so the same model can support classification metrics and
    downstream Scanpy handoff.
    """

    supported_tasks: tuple[str, ...] = ("classification",)
    supports_training = True

    def __init__(
        self,
        *,
        backbone: ScGPTBackbone,
        checkpoint_id: str,
        tuning_strategy: str,
        num_classes: int,
        label_categories: tuple[str, ...] | None = None,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.checkpoint_id = checkpoint_id
        self.tuning_strategy = tuning_strategy
        self.num_classes = num_classes
        self.label_categories = label_categories
        self.classifier_dropout = classifier_dropout
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(backbone.d_model),
            nn.Dropout(classifier_dropout),
            nn.Linear(backbone.d_model, num_classes),
        )

    def train(self, mode: bool = True) -> ScGPTAnnotationModel:
        super().train(mode)
        if mode and self.tuning_strategy == "head":
            self.backbone.eval()
        return self

    def _forward_outputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        pooled = self.backbone(
            batch["gene_ids"],
            batch["values"],
            batch["padding_mask"],
        )
        logits = self.classifier_head(pooled)
        latent = functional.normalize(pooled, p=2, dim=1)
        return {"logits": logits, "latent": latent}

    def predict_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict annotation logits and normalized embeddings for a token batch."""

        return self._forward_outputs(batch)

    def compute_task_loss(
        self,
        task_name: str,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
        """Compute the classification loss for scGPT annotation tuning."""

        if task_name != "classification":
            msg = f"ScGPT annotation model does not support task '{task_name}'."
            raise ValueError(msg)
        if "y" not in batch:
            msg = (
                "scGPT annotation fine-tuning requires encoded labels in each batch. "
                "Prepare data with label_key=... before calling Trainer.fit()."
            )
            raise ValueError(msg)
        outputs = self._forward_outputs(batch)
        loss = functional.cross_entropy(outputs["logits"], batch["y"])
        accuracy = (outputs["logits"].argmax(dim=1) == batch["y"]).float().mean()
        stats = {
            "loss": float(loss.detach().cpu()),
            "accuracy": float(accuracy.detach().cpu()),
        }
        return loss, stats, outputs


def _freeze_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _unfreeze_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def load_scgpt_annotation_model(
    *,
    num_classes: int,
    checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
    tuning_strategy: str = "head",
    label_categories: tuple[str, ...] | None = None,
    lora_config: ScGPTLoRAConfig | None = None,
    classifier_dropout: float = 0.1,
    device: str = "auto",
    cache_dir: str | Path | None = None,
) -> ScGPTAnnotationModel:
    """Load an experimental scGPT annotation model for ``Trainer``.

    Parameters
    ----------
    num_classes
        Number of annotation classes.
    checkpoint
        scGPT checkpoint identifier. ``"whole-human"`` is the only supported
        public checkpoint in ``v0.1.5``.
    tuning_strategy
        Either ``"head"`` or ``"lora"``.
    label_categories
        Optional class-name order used for reporting.
    lora_config
        Optional LoRA configuration. Defaults to :class:`ScGPTLoRAConfig`.
    classifier_dropout
        Dropout applied in the classification head.
    device
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    cache_dir
        Optional checkpoint cache root.

    Returns
    -------
    ScGPTAnnotationModel
        A model ready for ``Trainer(..., task="classification")``.

    Raises
    ------
    ValueError
        If the tuning strategy is unsupported.
    """

    if tuning_strategy not in {"head", "lora"}:
        msg = (
            "Unsupported tuning strategy "
            f"'{tuning_strategy}'. Expected one of: head, lora."
        )
        raise ValueError(msg)

    backbone, _, _ = _load_scgpt_backbone(checkpoint, cache_dir=cache_dir)
    _freeze_parameters(backbone)

    if tuning_strategy == "lora":
        apply_scgpt_lora(backbone, lora_config or ScGPTLoRAConfig())

    model = ScGPTAnnotationModel(
        backbone=backbone,
        checkpoint_id=checkpoint,
        tuning_strategy=tuning_strategy,
        num_classes=num_classes,
        label_categories=label_categories,
        classifier_dropout=classifier_dropout,
    )
    _unfreeze_parameters(model.classifier_head)
    resolved_device = resolve_device(device)
    return model.to(resolved_device)
