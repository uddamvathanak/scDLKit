"""Plain PyTorch training loop."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from scdlkit.data.datasets import AnnDataset
from scdlkit.data.schemas import SplitData
from scdlkit.tasks import BaseTask, get_task
from scdlkit.training.callbacks import EarlyStoppingState
from scdlkit.utils import resolve_device, set_seed


class Trainer:
    """Train scDLKit models with a task adapter."""

    def __init__(
        self,
        model: torch.nn.Module,
        task: str | BaseTask,
        *,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "auto",
        mixed_precision: bool = False,
        early_stopping_patience: int = 10,
        checkpoint: bool = True,
        seed: int = 42,
    ):
        self.model = model
        self.task = get_task(task) if isinstance(task, str) else task
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = resolve_device(device)
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint = checkpoint
        self.seed = seed
        self.history_: list[dict[str, float | int]] = []
        self.best_state_dict_: dict[str, Any] | None = None
        self.best_loss_: float | None = None
        self.best_epoch_: int | None = None
        supported_tasks = getattr(self.model, "supported_tasks", ())
        if supported_tasks and self.task.name not in supported_tasks:
            msg = (
                f"Model '{self.model.__class__.__name__}' does not support task "
                f"'{self.task.name}'."
            )
            raise ValueError(msg)
        self.model.to(self.device)

    @staticmethod
    def _coerce_dataset(
        dataset: SplitData | Dataset[dict[str, torch.Tensor]],
    ) -> Dataset[dict[str, torch.Tensor]]:
        if isinstance(dataset, SplitData):
            return AnnDataset(dataset)
        return dataset

    def fit(
        self,
        train_data: SplitData | Dataset[dict[str, torch.Tensor]],
        val_data: SplitData | Dataset[dict[str, torch.Tensor]] | None = None,
    ) -> Trainer:
        """Train the model and restore the best checkpointed state."""

        set_seed(self.seed)
        train_dataset = self._coerce_dataset(train_data)
        val_dataset = self._coerce_dataset(val_data) if val_data is not None else None
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            if val_dataset
            else None
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        stopper = EarlyStoppingState(best_loss=float("inf"), best_epoch=0)

        for epoch in tqdm(range(1, self.epochs + 1), desc="training", leave=False):
            train_metrics = self._run_epoch(train_loader, optimizer=optimizer, scaler=scaler)
            record: dict[str, float | int] = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, optimizer=None, scaler=None)
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
                monitor_loss = float(val_metrics["loss"])
            else:
                monitor_loss = float(train_metrics["loss"])
            self.history_.append(record)
            if monitor_loss < stopper.best_loss:
                stopper.best_loss = monitor_loss
                stopper.best_epoch = epoch
                stopper.epochs_without_improvement = 0
                self.best_loss_ = monitor_loss
                self.best_epoch_ = epoch
                if self.checkpoint:
                    self.best_state_dict_ = copy.deepcopy(self.model.state_dict())
            else:
                stopper.epochs_without_improvement += 1
                if stopper.epochs_without_improvement >= self.early_stopping_patience:
                    break

        if self.best_state_dict_ is not None:
            self.model.load_state_dict(self.best_state_dict_)
        return self

    def _run_epoch(
        self,
        loader: DataLoader[dict[str, torch.Tensor]],
        *,
        optimizer: torch.optim.Optimizer | None,
        scaler: torch.amp.GradScaler | None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)
        totals: dict[str, float] = {}
        num_batches = 0
        for batch in loader:
            device_batch = {key: value.to(self.device) for key, value in batch.items()}
            context = torch.autocast(
                device_type=self.device.type,
                enabled=self.mixed_precision,
            )
            with context:
                loss, stats, _ = self.task.compute_loss(self.model, device_batch)
            if training and optimizer is not None and scaler is not None:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            num_batches += 1
        return {key: value / max(num_batches, 1) for key, value in totals.items()}

    def predict_dataset(self, data: SplitData | Dataset[dict[str, torch.Tensor]]) -> dict[str, Any]:
        """Run inference on a dataset and collect batched outputs."""

        dataset = self._coerce_dataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        predictions: dict[str, list[torch.Tensor]] = {}
        with torch.inference_mode():
            for batch in loader:
                device_batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(device_batch["x"])
                for key, value in outputs.items():
                    predictions.setdefault(key, []).append(value.detach().cpu())
                predictions.setdefault("x", []).append(device_batch["x"].detach().cpu())
                if "y" in device_batch:
                    predictions.setdefault("y", []).append(device_batch["y"].detach().cpu())
                if "batch" in device_batch:
                    predictions.setdefault("batch", []).append(device_batch["batch"].detach().cpu())
        return {key: torch.cat(chunks).numpy() for key, chunks in predictions.items()}

    @property
    def history_frame(self) -> pd.DataFrame:
        """Training history as a DataFrame."""

        return pd.DataFrame(self.history_)

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist the best model checkpoint to disk."""

        if self.best_state_dict_ is None:
            msg = "No checkpoint is available. Train the model first."
            raise RuntimeError(msg)
        checkpoint_path = Path(path)
        torch.save(self.best_state_dict_, checkpoint_path)
        return checkpoint_path
