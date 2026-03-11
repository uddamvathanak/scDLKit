"""Model registry and implementations."""

from scdlkit.models.autoencoder import AutoEncoder
from scdlkit.models.base import BaseModel
from scdlkit.models.classifier import MLPClassifier
from scdlkit.models.denoising import DenoisingAutoEncoder
from scdlkit.models.registry import create_model, register_model
from scdlkit.models.transformer import TransformerAutoEncoder
from scdlkit.models.vae import VariationalAutoEncoder

__all__ = [
    "AutoEncoder",
    "BaseModel",
    "DenoisingAutoEncoder",
    "MLPClassifier",
    "TransformerAutoEncoder",
    "VariationalAutoEncoder",
    "create_model",
    "register_model",
]
