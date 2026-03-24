"""Experimental scGPT annotation fine-tuning helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from torch import Tensor, nn
from torch.nn import functional

from scdlkit.foundation.adapters import apply_scgpt_adapters
from scdlkit.foundation.cache import DEFAULT_SCGPT_CHECKPOINT
from scdlkit.foundation.ia3 import apply_scgpt_ia3
from scdlkit.foundation.lora import ScGPTLoRAConfig, apply_scgpt_lora
from scdlkit.foundation.peft import (
    AdapterConfig,
    AnnotationStrategy,
    IA3Config,
    LoRAConfig,
    PEFTConfig,
    PrefixTuningConfig,
    default_strategy_config,
)
from scdlkit.foundation.prefix_tuning import apply_scgpt_prefix_tuning
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
        One of ``"head"``, ``"full_finetune"``, ``"lora"``, ``"adapter"``,
        ``"prefix_tuning"``, or ``"ia3"``.
    num_classes
        Number of annotation classes.
    label_categories
        Optional class-name order used for reporting and confusion matrices.
    classifier_dropout
        Dropout applied before the final classification layer.
    strategy_config
        Optional PEFT config used for manifest metadata and reloads.

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
        backbone: ScGPTBackbone | nn.Module,
        checkpoint_id: str,
        tuning_strategy: AnnotationStrategy,
        num_classes: int,
        label_categories: tuple[str, ...] | None = None,
        classifier_dropout: float = 0.1,
        strategy_config: PEFTConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.checkpoint_id = checkpoint_id
        self.tuning_strategy = tuning_strategy
        self.num_classes = num_classes
        self.label_categories = label_categories
        self.classifier_dropout = classifier_dropout
        self.strategy_config = strategy_config
        d_model = int(cast(Any, backbone).d_model)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model, num_classes),
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
            raise ValueError(f"ScGPT annotation model does not support task '{task_name}'.")
        if "y" not in batch:
            raise ValueError(
                "scGPT annotation fine-tuning requires encoded labels in each batch. "
                "Prepare data with label_key=... before calling Trainer.fit()."
            )
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


def _resolve_strategy_config(
    *,
    tuning_strategy: AnnotationStrategy,
    strategy_config: PEFTConfig | None,
    lora_config: LoRAConfig | None,
) -> PEFTConfig | None:
    if lora_config is not None and strategy_config is not None:
        raise ValueError("Pass either strategy_config or lora_config, not both.")
    if tuning_strategy == "lora":
        resolved = strategy_config or lora_config or default_strategy_config("lora")
        if resolved is not None and not isinstance(resolved, LoRAConfig):
            raise ValueError("LoRA tuning requires a LoRAConfig strategy config.")
        return resolved
    if lora_config is not None:
        raise ValueError("lora_config is only valid when tuning_strategy='lora'.")
    resolved = strategy_config or default_strategy_config(tuning_strategy)
    if (
        tuning_strategy == "adapter"
        and resolved is not None
        and not isinstance(resolved, AdapterConfig)
    ):
        raise ValueError("Adapter tuning requires an AdapterConfig strategy config.")
    if (
        tuning_strategy == "prefix_tuning"
        and resolved is not None
        and not isinstance(resolved, PrefixTuningConfig)
    ):
        raise ValueError("Prefix tuning requires a PrefixTuningConfig strategy config.")
    if tuning_strategy == "ia3" and resolved is not None and not isinstance(resolved, IA3Config):
        raise ValueError("IA3 tuning requires an IA3Config strategy config.")
    return resolved


def _apply_tuning_strategy(
    backbone: ScGPTBackbone,
    *,
    tuning_strategy: AnnotationStrategy,
    strategy_config: PEFTConfig | None,
) -> PEFTConfig | None:
    _freeze_parameters(backbone)
    if tuning_strategy == "head":
        return None
    if tuning_strategy == "full_finetune":
        _unfreeze_parameters(backbone)
        return None
    if tuning_strategy == "lora":
        config = strategy_config or default_strategy_config("lora")
        assert isinstance(config, LoRAConfig)
        apply_scgpt_lora(backbone, config)
        return config
    if tuning_strategy == "adapter":
        config = strategy_config or default_strategy_config("adapter")
        assert isinstance(config, AdapterConfig)
        apply_scgpt_adapters(backbone, config)
        return config
    if tuning_strategy == "prefix_tuning":
        config = strategy_config or default_strategy_config("prefix_tuning")
        assert isinstance(config, PrefixTuningConfig)
        apply_scgpt_prefix_tuning(backbone, config)
        return config
    if tuning_strategy == "ia3":
        config = strategy_config or default_strategy_config("ia3")
        assert isinstance(config, IA3Config)
        apply_scgpt_ia3(backbone, config)
        return config
    raise ValueError(
        "Unsupported tuning strategy "
        f"'{tuning_strategy}'. Expected one of: head, full_finetune, lora, "
        "adapter, prefix_tuning, ia3."
    )


def load_scgpt_annotation_model(
    *,
    num_classes: int,
    checkpoint: str = DEFAULT_SCGPT_CHECKPOINT,
    tuning_strategy: AnnotationStrategy = "head",
    label_categories: tuple[str, ...] | None = None,
    strategy_config: PEFTConfig | None = None,
    lora_config: ScGPTLoRAConfig | LoRAConfig | None = None,
    classifier_dropout: float = 0.1,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    preloaded_state_dict: dict | None = None,
) -> ScGPTAnnotationModel:
    """Load an experimental scGPT annotation model for ``Trainer``.

    Parameters
    ----------
    num_classes
        Number of annotation classes.
    checkpoint
        scGPT checkpoint identifier. ``"whole-human"`` is the only supported
        public checkpoint in the current experimental release line.
    tuning_strategy
        One of ``"head"``, ``"full_finetune"``, ``"lora"``, ``"adapter"``,
        ``"prefix_tuning"``, or ``"ia3"``.
    label_categories
        Optional class-name order used for reporting.
    strategy_config
        Optional strategy-specific PEFT configuration.
    lora_config
        Backward-compatible alias for the LoRA strategy config.
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
        If the tuning strategy or strategy config is unsupported.
    """

    resolved_strategy_config = _resolve_strategy_config(
        tuning_strategy=tuning_strategy,
        strategy_config=strategy_config,
        lora_config=lora_config,
    )
    backbone, _, _ = _load_scgpt_backbone(
        checkpoint, cache_dir=cache_dir, preloaded_state_dict=preloaded_state_dict
    )
    resolved_strategy_config = _apply_tuning_strategy(
        backbone,
        tuning_strategy=tuning_strategy,
        strategy_config=resolved_strategy_config,
    )

    model = ScGPTAnnotationModel(
        backbone=backbone,
        checkpoint_id=checkpoint,
        tuning_strategy=tuning_strategy,
        num_classes=num_classes,
        label_categories=label_categories,
        classifier_dropout=classifier_dropout,
        strategy_config=resolved_strategy_config,
    )
    _unfreeze_parameters(model.classifier_head)
    resolved_device = resolve_device(device)
    return model.to(resolved_device)


__all__ = [
    "ScGPTAnnotationModel",
    "ScGPTLoRAConfig",
    "load_scgpt_annotation_model",
]
