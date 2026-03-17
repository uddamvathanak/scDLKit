"""High-level orchestration API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from anndata import AnnData

from scdlkit.data import PreparedData, prepare_data, transform_adata
from scdlkit.data.schemas import SplitData
from scdlkit.evaluation import evaluate_predictions
from scdlkit.evaluation.report import save_markdown_report, save_metrics_table
from scdlkit.models import BaseModel, create_model
from scdlkit.tasks import get_task
from scdlkit.training import Trainer
from scdlkit.utils import ensure_directory
from scdlkit.visualization import (
    plot_confusion_matrix,
    plot_latent_embedding,
    plot_losses,
    plot_reconstruction_scatter,
)


class TaskRunner:
    """Beginner-facing training, evaluation, and visualization workflow.

    `TaskRunner` is the highest-level stable entry point in scDLKit. It owns the
    common AnnData workflow:

    1. prepare and split an :class:`~anndata.AnnData`
    2. create or accept a model
    3. train with a task adapter
    4. evaluate and visualize outputs
    5. hand embeddings or reconstructions back to the user

    Parameters
    ----------
    model
        Built-in model name or an instantiated scDLKit model.
    task
        One of ``"representation"``, ``"reconstruction"``, or ``"classification"``.
    latent_dim
        Latent dimensionality for encoder-based built-in models.
    hidden_dims
        Hidden-layer sizes for built-in feed-forward models.
    epochs
        Maximum number of training epochs.
    batch_size
        Training and inference batch size.
    lr
        Optimizer learning rate.
    device
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    mixed_precision
        Enable AMP on CUDA when supported.
    early_stopping_patience
        Number of epochs without improvement before stopping early.
    checkpoint
        Whether to keep and restore the best validation checkpoint.
    seed
        Random seed used during training.
    layer
        AnnData matrix layer to read. ``"X"`` uses ``adata.X``.
    use_hvg
        Whether to select highly variable genes during preparation.
    n_top_genes
        Number of highly variable genes to retain when ``use_hvg=True``.
    normalize
        Whether to run Scanpy total-count normalization.
    log1p
        Whether to run Scanpy ``log1p`` transformation.
    scale
        Whether to standardize features.
    label_key
        Observation column used for labels and evaluation.
    batch_key
        Observation column used for batch-aware splitting and optional metrics.
    val_size
        Validation split fraction.
    test_size
        Test split fraction.
    batch_aware_split
        Whether to keep batch groups together when splitting.
    random_state
        Random state for deterministic data splitting.
    output_dir
        Optional directory for saved checkpoints and reports.
    model_kwargs
        Extra keyword arguments forwarded to built-in model construction.

    Notes
    -----
    Use :class:`scdlkit.training.Trainer` directly when you need lower-level
    control or when wrapping a custom module with the adapter APIs.

    Examples
    --------
    >>> import scanpy as sc
    >>> from scdlkit import TaskRunner
    >>> adata = sc.datasets.pbmc3k_processed()
    >>> runner = TaskRunner(
    ...     model="vae",
    ...     task="representation",
    ...     label_key="louvain",
    ...     device="auto",
    ...     epochs=20,
    ...     batch_size=128,
    ...     model_kwargs={"kl_weight": 1e-3},
    ... )
    >>> runner.fit(adata)
    >>> latent = runner.encode(adata)
    """

    def __init__(
        self,
        *,
        model: str | BaseModel,
        task: str,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "auto",
        mixed_precision: bool = False,
        early_stopping_patience: int = 10,
        checkpoint: bool = True,
        seed: int = 42,
        layer: str = "X",
        use_hvg: bool = False,
        n_top_genes: int = 2000,
        normalize: bool = False,
        log1p: bool = False,
        scale: bool = False,
        label_key: str | None = None,
        batch_key: str | None = None,
        val_size: float = 0.15,
        test_size: float = 0.15,
        batch_aware_split: bool = False,
        random_state: int = 42,
        output_dir: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        self.model_spec = model
        self.task_name = task
        self.task = get_task(task)
        self.model_kwargs = model_kwargs or {}
        self.model_defaults: dict[str, Any] = {
            "latent_dim": latent_dim,
            "hidden_dims": hidden_dims,
        }
        self.prepare_kwargs: dict[str, Any] = {
            "layer": layer,
            "use_hvg": use_hvg,
            "n_top_genes": n_top_genes,
            "normalize": normalize,
            "log1p": log1p,
            "scale": scale,
            "label_key": label_key,
            "batch_key": batch_key,
            "val_size": val_size,
            "test_size": test_size,
            "batch_aware_split": batch_aware_split,
            "random_state": random_state,
        }
        self.training_kwargs: dict[str, Any] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "mixed_precision": mixed_precision,
            "early_stopping_patience": early_stopping_patience,
            "checkpoint": checkpoint,
            "seed": seed,
        }
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.prepared_data_: PreparedData | None = None
        self.model_: BaseModel | None = model if isinstance(model, BaseModel) else None
        self.trainer_: Trainer | None = None
        self.metrics_: dict[str, Any] | None = None
        self.last_predictions_: dict[str, np.ndarray] | None = None

    def _create_model(self, input_dim: int, num_classes: int | None) -> BaseModel:
        if isinstance(self.model_spec, BaseModel):
            model = self.model_spec
        else:
            kwargs: dict[str, Any] = {
                "input_dim": input_dim,
                **self.model_defaults,
                **self.model_kwargs,
            }
            if self.model_spec == "mlp_classifier":
                if num_classes is None:
                    msg = "Classification requires label_key and encoded classes."
                    raise ValueError(msg)
                kwargs.pop("latent_dim", None)
                kwargs["num_classes"] = num_classes
            model = create_model(self.model_spec, **kwargs)
        if self.task_name not in getattr(model, "supported_tasks", ()):
            msg = f"Model '{model.__class__.__name__}' does not support task '{self.task_name}'."
            raise ValueError(msg)
        return model

    def _prepare_for_inference(self, adata: AnnData) -> SplitData:
        if self.prepared_data_ is None:
            msg = "TaskRunner must be fit before inference."
            raise RuntimeError(msg)
        return transform_adata(
            adata,
            self.prepared_data_.preprocessing,
            label_encoder=self.prepared_data_.label_encoder,
            batch_encoder=self.prepared_data_.batch_encoder,
        )

    def fit(
        self,
        adata: AnnData,
        *,
        val_adata: AnnData | None = None,
        test_adata: AnnData | None = None,
    ) -> TaskRunner:
        """Prepare data, instantiate the model, and train it.

        Parameters
        ----------
        adata
            Primary AnnData used to create train, validation, and test splits.
        val_adata
            Optional external validation split prepared with the same preprocessing.
        test_adata
            Optional external test split prepared with the same preprocessing.

        Returns
        -------
        TaskRunner
            The fitted runner.
        """

        prepared = prepare_data(adata, **self.prepare_kwargs)
        if val_adata is not None:
            prepared.val = self._prepare_external_split(val_adata, prepared)
        if test_adata is not None:
            prepared.test = self._prepare_external_split(test_adata, prepared)
        num_classes = len(prepared.label_encoder) if prepared.label_encoder is not None else None
        model = self._create_model(prepared.input_dim, num_classes)
        trainer = Trainer(model=model, task=self.task_name, **self.training_kwargs)
        trainer.fit(prepared.train, prepared.val)
        self.prepared_data_ = prepared
        self.model_ = model
        self.trainer_ = trainer
        if self.output_dir is not None and trainer.best_state_dict_ is not None:
            directory = ensure_directory(self.output_dir)
            trainer.save_checkpoint(directory / "best_model.pt")
        return self

    def _prepare_external_split(self, adata: AnnData, prepared: PreparedData) -> SplitData:
        return transform_adata(
            adata,
            prepared.preprocessing,
            label_encoder=prepared.label_encoder,
            batch_encoder=prepared.batch_encoder,
        )

    def _run_predictions(self, split: SplitData) -> dict[str, np.ndarray]:
        if self.trainer_ is None:
            msg = "TaskRunner must be fit before evaluation."
            raise RuntimeError(msg)
        return self.trainer_.predict_dataset(split)

    def evaluate(self, adata: AnnData | None = None) -> dict[str, Any]:
        """Evaluate the current model on a held-out split or a provided AnnData object.

        Parameters
        ----------
        adata
            Optional AnnData to transform and evaluate. When omitted, the runner
            evaluates the test split, then validation split, then train split.

        Returns
        -------
        dict[str, Any]
            Task-specific metrics such as reconstruction, representation, or
            classification scores.
        """

        if self.prepared_data_ is None:
            msg = "TaskRunner must be fit before evaluation."
            raise RuntimeError(msg)
        split = self._prepare_for_inference(adata) if adata is not None else None
        if split is None:
            split = self.prepared_data_.test or self.prepared_data_.val or self.prepared_data_.train
        predictions = self._run_predictions(split)
        self.last_predictions_ = predictions
        self.metrics_ = evaluate_predictions(self.task_name, predictions)
        return self.metrics_

    def predict(self, adata: AnnData) -> np.ndarray:
        """Run task-dependent prediction on new AnnData.

        Parameters
        ----------
        adata
            AnnData object to transform and run through the fitted model.

        Returns
        -------
        numpy.ndarray
            For classification tasks, class predictions. For reconstruction-capable
            tasks, reconstructed expression values.

        Notes
        -----
        This method is intentionally backward compatible, but its return type is
        task-dependent. For reconstruction-capable models, prefer
        :meth:`reconstruct` in new tutorials and user-facing code.
        """

        predictions = self._run_predictions(self._prepare_for_inference(adata))
        if self.task_name == "classification":
            return predictions["logits"].argmax(axis=1)
        return predictions["reconstruction"]

    def reconstruct(self, adata: AnnData) -> np.ndarray:
        """Return reconstructed or predicted gene-expression values.

        Parameters
        ----------
        adata
            AnnData object to transform and run through the fitted model.

        Returns
        -------
        numpy.ndarray
            Reconstructed expression values for reconstruction-capable models.

        Raises
        ------
        ValueError
            If the fitted task is classification-only and does not expose
            reconstructed expression outputs.
        """

        if self.task_name == "classification":
            msg = "Classification models do not expose reconstructed expression outputs."
            raise ValueError(msg)
        predictions = self._run_predictions(self._prepare_for_inference(adata))
        return predictions["reconstruction"]

    def encode(self, adata: AnnData) -> np.ndarray:
        """Encode new AnnData into latent representations.

        Parameters
        ----------
        adata
            AnnData object to transform and encode with the fitted model.

        Returns
        -------
        numpy.ndarray
            Latent embedding matrix suitable for storage in ``adata.obsm``.

        Raises
        ------
        ValueError
            If the task does not expose latent encodings.
        """

        if self.task_name == "classification":
            msg = "Classification models do not expose latent encodings in v0.1."
            raise ValueError(msg)
        predictions = self._run_predictions(self._prepare_for_inference(adata))
        return predictions["latent"]

    def plot_losses(self) -> tuple[Any, Any]:
        if self.trainer_ is None:
            msg = "TaskRunner must be fit before plotting losses."
            raise RuntimeError(msg)
        return plot_losses(self.trainer_.history_frame)

    def plot_latent(self, *, method: str = "umap", color: str | None = None) -> tuple[Any, Any]:
        if self.prepared_data_ is None:
            msg = "TaskRunner must be fit before plotting latent embeddings."
            raise RuntimeError(msg)
        if self.task_name == "classification":
            msg = "Latent plotting is only available for encoder-based tasks."
            raise ValueError(msg)
        split = self.prepared_data_.test or self.prepared_data_.val or self.prepared_data_.train
        predictions = self._run_predictions(split)
        color_values = None
        label_key = self.prepare_kwargs["label_key"]
        batch_key = self.prepare_kwargs["batch_key"]
        if color in {"label", label_key} and split.labels is not None:
            color_values = split.labels
        elif color in {"batch", batch_key} and split.batches is not None:
            color_values = split.batches
        return plot_latent_embedding(predictions["latent"], color=color_values, method=method)

    def plot_reconstruction(self, feature: str | int | None = None) -> tuple[Any, Any]:
        if self.prepared_data_ is None:
            msg = "TaskRunner must be fit before plotting reconstructions."
            raise RuntimeError(msg)
        if self.task_name == "classification":
            msg = "Reconstruction plotting is only available for encoder-based tasks."
            raise ValueError(msg)
        split = self.prepared_data_.test or self.prepared_data_.val or self.prepared_data_.train
        predictions = self._run_predictions(split)
        feature_index = 0
        feature_name = None
        if isinstance(feature, str):
            feature_index = self.prepared_data_.feature_names.index(feature)
            feature_name = feature
        elif isinstance(feature, int):
            feature_index = feature
            feature_name = self.prepared_data_.feature_names[feature]
        return plot_reconstruction_scatter(
            predictions["x"],
            predictions["reconstruction"],
            feature_index=feature_index,
            feature_name=feature_name,
        )

    def plot_confusion_matrix(self) -> tuple[Any, Any]:
        metrics = self.metrics_ or self.evaluate()
        confusion = metrics.get("confusion_matrix")
        if confusion is None:
            msg = "Confusion matrix is only available for classification tasks."
            raise ValueError(msg)
        class_names = (
            list(self.prepared_data_.label_encoder)
            if self.prepared_data_ and self.prepared_data_.label_encoder
            else None
        )
        return plot_confusion_matrix(confusion, class_names=class_names)

    def save_report(self, path: str | Path) -> Path:
        """Write a Markdown report and scalar-metric CSV.

        Parameters
        ----------
        path
            Markdown output path. A sibling ``.csv`` file is written next to it.

        Returns
        -------
        pathlib.Path
            The resolved Markdown report path.
        """

        if self.metrics_ is None:
            self.evaluate()
        if self.metrics_ is None or self.trainer_ is None:
            msg = "TaskRunner must be fit before saving reports."
            raise RuntimeError(msg)
        report_path = Path(path)
        if report_path.parent != Path("."):
            ensure_directory(report_path.parent)
        csv_path = report_path.with_suffix(".csv")
        title = f"scDLKit report: {self.task_name}"
        save_markdown_report(self.metrics_, path=report_path, title=title)
        save_metrics_table(self.metrics_, csv_path)
        return report_path
