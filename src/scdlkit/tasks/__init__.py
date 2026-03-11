"""Task registry and implementations."""

from scdlkit.tasks.base import BaseTask, get_task
from scdlkit.tasks.classification import ClassificationTask
from scdlkit.tasks.reconstruction import ReconstructionTask
from scdlkit.tasks.representation import RepresentationTask

__all__ = [
    "BaseTask",
    "ClassificationTask",
    "ReconstructionTask",
    "RepresentationTask",
    "get_task",
]
