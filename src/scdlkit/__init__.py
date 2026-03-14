"""Public package surface for scDLKit."""

from scdlkit.data import PreparedData, prepare_data
from scdlkit.evaluation.compare import BenchmarkResult, compare_models
from scdlkit.models import BaseModel, create_model
from scdlkit.runner import TaskRunner
from scdlkit.training import Trainer

__all__ = [
    "BaseModel",
    "BenchmarkResult",
    "PreparedData",
    "TaskRunner",
    "Trainer",
    "compare_models",
    "create_model",
    "prepare_data",
]

__version__ = "0.1.2"
