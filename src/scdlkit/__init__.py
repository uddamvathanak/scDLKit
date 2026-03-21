"""Public package surface for scDLKit.

The package exposes two main public routes:

- stable baseline workflows through :class:`TaskRunner`
- experimental labeled-annotation adaptation through
  :func:`adapt_annotation` and :class:`AnnotationRunner`
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from anndata import AnnData

from scdlkit.data import PreparedData, prepare_data
from scdlkit.evaluation.compare import BenchmarkResult, compare_models
from scdlkit.models import BaseModel, create_model
from scdlkit.runner import TaskRunner
from scdlkit.training import Trainer

if TYPE_CHECKING:
    from scdlkit.foundation import ScGPTAnnotationDataReport

_DEFAULT_ANNOTATION_CHECKPOINT = "whole-human"
_FoundationRunnerBase: type[Any] = object
_FOUNDATION_IMPORT_ERROR: ImportError | None = None

try:  # pragma: no cover - foundation extras are covered in dedicated tests
    from scdlkit.foundation import ScGPTAnnotationRunner as _ImportedAnnotationRunner
except ImportError as exc:  # pragma: no cover - exercised in minimal install smoke
    _FOUNDATION_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in foundation-enabled tests
    _FoundationRunnerBase = _ImportedAnnotationRunner


class AnnotationRunner(_FoundationRunnerBase):
    """Experimental top-level alias for the scGPT annotation wrapper.

    Use this class when you want the step-by-step version of the beginner
    annotation workflow:

    1. inspect a labeled dataset
    2. compare frozen and tuned strategies
    3. annotate ``AnnData`` in place
    4. save and reload the fitted runner

    Notes
    -----
    This class is experimental. In the current release line it is backed only by
    the scGPT ``whole-human`` annotation path for labeled human scRNA-seq
    datasets.
    """

    if _FOUNDATION_IMPORT_ERROR is not None:  # pragma: no cover - minimal install guard
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            msg = (
                "Experimental annotation support requires the foundation extra. "
                'Install scdlkit with `pip install "scdlkit[foundation]"`.'
            )
            raise ImportError(msg) from _FOUNDATION_IMPORT_ERROR

        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> Any:
            msg = (
                "Experimental annotation support requires the foundation extra. "
                'Install scdlkit with `pip install "scdlkit[foundation]"`.'
            )
            raise ImportError(msg) from _FOUNDATION_IMPORT_ERROR


def inspect_annotation_data(
    adata: AnnData,
    *,
    label_key: str,
    checkpoint: str = _DEFAULT_ANNOTATION_CHECKPOINT,
    use_raw: bool = True,
    min_gene_overlap: int = 500,
    min_cells_per_class: int = 10,
) -> ScGPTAnnotationDataReport:
    """Inspect a labeled dataset for experimental annotation adaptation.

    Parameters
    ----------
    adata
        Human single-cell ``AnnData`` to inspect.
    label_key
        Required label column in ``adata.obs``.
    checkpoint
        Experimental annotation checkpoint identifier. The public route
        currently supports only ``"whole-human"``.
    use_raw
        Use ``adata.raw`` when available.
    min_gene_overlap
        Warning threshold for matched genes against the checkpoint vocabulary.
    min_cells_per_class
        Warning threshold for the smallest label class.

    Returns
    -------
    ScGPTAnnotationDataReport
        A lightweight compatibility report with overlap, class-balance, and
        stratification checks.

    Raises
    ------
    ValueError
        If labels are missing or the expression matrix is incompatible with the
        current experimental scGPT path.

    Notes
    -----
    This function is experimental. It currently routes only to the scGPT
    ``whole-human`` annotation preparation path for human scRNA-seq data.

    Examples
    --------
    >>> report = inspect_annotation_data(adata, label_key="cell_type")
    >>> report.num_genes_matched > 0
    True
    """

    from scdlkit.foundation import inspect_scgpt_annotation_data

    return inspect_scgpt_annotation_data(
        adata,
        label_key=label_key,
        checkpoint=checkpoint,
        use_raw=use_raw,
        min_gene_overlap=min_gene_overlap,
        min_cells_per_class=min_cells_per_class,
    )


def adapt_annotation(
    adata: AnnData,
    *,
    label_key: str,
    checkpoint: str = _DEFAULT_ANNOTATION_CHECKPOINT,
    strategies: tuple[str, ...] = ("frozen_probe", "head"),
    batch_size: int = 64,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    device: str = "auto",
    output_dir: str | Path | None = None,
) -> AnnotationRunner:
    """Run the experimental beginner annotation quickstart in one call.

    Parameters
    ----------
    adata
        Labeled human single-cell ``AnnData`` to adapt on.
    label_key
        Required target label column in ``adata.obs``.
    checkpoint
        Experimental annotation checkpoint identifier. The public route
        currently supports only ``"whole-human"``.
    strategies
        Strategy ladder to compare. The default quickstart uses
        ``("frozen_probe", "head")``. Add ``"lora"`` explicitly when needed.
    batch_size
        Inference and training batch size for the wrapper workflow.
    val_size
        Validation split fraction.
    test_size
        Test split fraction.
    random_state
        Random seed used for splitting and trainable strategies.
    device
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    output_dir
        Optional artifact directory for reports, plots, and saved runner state.

    Returns
    -------
    AnnotationRunner
        A fitted experimental runner holding the best strategy and wrapper
        summary.

    Raises
    ------
    ImportError
        If the package was installed without ``scdlkit[foundation]``.
    ValueError
        If labels, checkpoint compatibility, or gene overlap checks fail.

    Notes
    -----
    This function is experimental. It currently routes only to the scGPT
    ``whole-human`` annotation path for labeled human scRNA-seq data.

    Examples
    --------
    >>> runner = adapt_annotation(adata, label_key="cell_type")
    >>> runner.best_strategy_ in {"frozen_probe", "head", "lora"}
    True
    """

    runner = AnnotationRunner(
        label_key=label_key,
        checkpoint=checkpoint,
        strategies=strategies,
        batch_size=batch_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        device=device,
        output_dir=output_dir,
    )
    runner.fit_compare(adata)
    return runner


__all__ = [
    "AnnotationRunner",
    "BaseModel",
    "BenchmarkResult",
    "PreparedData",
    "TaskRunner",
    "Trainer",
    "adapt_annotation",
    "compare_models",
    "create_model",
    "inspect_annotation_data",
    "prepare_data",
]

__version__ = "0.1.5"
