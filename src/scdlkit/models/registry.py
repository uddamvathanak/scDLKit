"""Model registry helpers."""

from __future__ import annotations

from collections.abc import Callable

from scdlkit.models.base import BaseModel

ModelFactory = Callable[..., BaseModel]
_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str, *aliases: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a model factory under one or more names."""

    def decorator(factory: ModelFactory) -> ModelFactory:
        for key in (name, *aliases):
            _REGISTRY[key] = factory
        return factory

    return decorator


def create_model(name: str, **kwargs: object) -> BaseModel:
    """Instantiate a registered model by name."""

    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        msg = f"Unknown model '{name}'. Available models: {available}"
        raise ValueError(msg) from exc
    return factory(**kwargs)
