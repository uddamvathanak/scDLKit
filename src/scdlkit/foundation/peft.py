"""Generic PEFT configuration helpers for experimental foundation adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

AnnotationStrategy = Literal[
    "frozen_probe",
    "head",
    "full_finetune",
    "lora",
    "adapter",
    "prefix_tuning",
    "ia3",
]

PEFT_STRATEGIES: tuple[AnnotationStrategy, ...] = (
    "lora",
    "adapter",
    "prefix_tuning",
    "ia3",
)

TRAINABLE_ANNOTATION_STRATEGIES: tuple[AnnotationStrategy, ...] = (
    "head",
    "full_finetune",
    *PEFT_STRATEGIES,
)

_LORA_TARGET_MODULES = ("out_proj", "linear1", "linear2")
_IA3_TARGET_MODULES = ("out_proj", "linear1")


@dataclass(frozen=True, slots=True)
class PEFTConfig:
    """Base class for experimental PEFT configuration objects."""

    config_type: str = "peft"

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_type"] = self.config_type
        return payload


@dataclass(frozen=True, slots=True)
class LoRAConfig(PEFTConfig):
    """Configuration for low-rank adaptation."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = _LORA_TARGET_MODULES
    config_type: str = "lora"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in the range [0, 1).")
        invalid = sorted(set(self.target_modules) - set(_LORA_TARGET_MODULES))
        if invalid:
            raise ValueError(
                "Unsupported LoRA target modules: "
                f"{', '.join(invalid)}. Supported values are "
                f"{', '.join(_LORA_TARGET_MODULES)}."
            )


@dataclass(frozen=True, slots=True)
class AdapterConfig(PEFTConfig):
    """Configuration for bottleneck adapter tuning."""

    bottleneck_dim: int = 64
    dropout: float = 0.05
    activation: Literal["relu", "gelu"] = "gelu"
    config_type: str = "adapter"

    def __post_init__(self) -> None:
        if self.bottleneck_dim <= 0:
            raise ValueError("Adapter bottleneck_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Adapter dropout must be in the range [0, 1).")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("Adapter activation must be 'relu' or 'gelu'.")


@dataclass(frozen=True, slots=True)
class PrefixTuningConfig(PEFTConfig):
    """Configuration for prefix-tuning style trainable prompts."""

    prefix_length: int = 20
    dropout: float = 0.05
    init_std: float = 0.02
    config_type: str = "prefix_tuning"

    def __post_init__(self) -> None:
        if self.prefix_length <= 0:
            raise ValueError("Prefix length must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Prefix tuning dropout must be in the range [0, 1).")
        if self.init_std <= 0:
            raise ValueError("Prefix tuning init_std must be positive.")


@dataclass(frozen=True, slots=True)
class IA3Config(PEFTConfig):
    """Configuration for IA3-style multiplicative scaling."""

    init_scale: float = 1.0
    target_modules: tuple[str, ...] = _IA3_TARGET_MODULES
    config_type: str = "ia3"

    def __post_init__(self) -> None:
        if self.init_scale <= 0:
            raise ValueError("IA3 init_scale must be positive.")
        invalid = sorted(set(self.target_modules) - set(_IA3_TARGET_MODULES))
        if invalid:
            raise ValueError(
                "Unsupported IA3 target modules: "
                f"{', '.join(invalid)}. Supported values are "
                f"{', '.join(_IA3_TARGET_MODULES)}."
            )


def default_strategy_config(strategy: AnnotationStrategy) -> PEFTConfig | None:
    """Return the default PEFT configuration for a supported strategy."""

    if strategy == "lora":
        return LoRAConfig()
    if strategy == "adapter":
        return AdapterConfig()
    if strategy == "prefix_tuning":
        return PrefixTuningConfig()
    if strategy == "ia3":
        return IA3Config()
    return None


def _config_class_for_type(config_type: str) -> type[PEFTConfig]:
    mapping: dict[str, type[PEFTConfig]] = {
        "lora": LoRAConfig,
        "adapter": AdapterConfig,
        "prefix_tuning": PrefixTuningConfig,
        "ia3": IA3Config,
    }
    try:
        return mapping[config_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported PEFT config type '{config_type}'.") from exc


def deserialize_peft_config(payload: Mapping[str, Any] | None) -> PEFTConfig | None:
    """Build a PEFT config instance from a serialized payload."""

    if payload is None:
        return None
    config_type = str(payload.get("config_type", ""))
    config_class = _config_class_for_type(config_type)
    kwargs = {key: value for key, value in payload.items() if key != "config_type"}
    return config_class(**kwargs)


def serialize_strategy_configs(
    strategy_configs: Mapping[str, PEFTConfig] | None,
) -> dict[str, dict[str, Any]] | None:
    """Serialize a per-strategy config mapping for manifests."""

    if not strategy_configs:
        return None
    return {str(strategy): config.to_payload() for strategy, config in strategy_configs.items()}


def deserialize_strategy_configs(
    payload: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, PEFTConfig]:
    """Deserialize a manifest strategy-config mapping."""

    if not payload:
        return {}
    resolved: dict[str, PEFTConfig] = {}
    for strategy, config in payload.items():
        deserialized = deserialize_peft_config(config)
        if deserialized is not None:
            resolved[str(strategy)] = deserialized
    return resolved


def resolve_strategy_configs(
    *,
    strategies: tuple[str, ...],
    strategy_configs: Mapping[str, PEFTConfig] | None = None,
    lora_config: LoRAConfig | None = None,
) -> dict[str, PEFTConfig]:
    """Normalize per-strategy configuration inputs for annotation workflows."""

    if lora_config is not None and strategy_configs is not None:
        raise ValueError("Pass either strategy_configs or lora_config, not both.")
    resolved = dict(strategy_configs or {})
    if lora_config is not None:
        resolved["lora"] = lora_config
    unknown = sorted(set(resolved) - set(strategies))
    if unknown:
        raise ValueError(
            f"Received strategy configs for strategies that are not enabled: {', '.join(unknown)}."
        )
    for strategy in strategies:
        default_config = default_strategy_config(strategy)  # type: ignore[arg-type]
        if default_config is not None:
            resolved.setdefault(strategy, default_config)
    return resolved


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count the trainable parameters of a module."""

    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


__all__ = [
    "AdapterConfig",
    "AnnotationStrategy",
    "IA3Config",
    "LoRAConfig",
    "PEFTConfig",
    "PEFT_STRATEGIES",
    "PrefixTuningConfig",
    "TRAINABLE_ANNOTATION_STRATEGIES",
    "count_trainable_parameters",
    "default_strategy_config",
    "deserialize_peft_config",
    "deserialize_strategy_configs",
    "resolve_strategy_configs",
    "serialize_strategy_configs",
]
