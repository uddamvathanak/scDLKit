"""Representation learning task adapter."""

from __future__ import annotations

from scdlkit.tasks.base import register_task
from scdlkit.tasks.reconstruction import ReconstructionTask


class RepresentationTask(ReconstructionTask):
    name = "representation"
    metric_group = "representation"


register_task(RepresentationTask())
