"""Result summary rows for the single-dataset/method AutoML stages (04/05/06).

The column order is the same in all stages (04 ``random_forest_baseline*``,
05 ``autogluon_benchmark_single``, 06 ``flaml_benchmark*``). The ``*_method`` variants add the
trailing ``method`` and ``percentage`` columns. The multi-row frames of the
``domain_adaptation`` scripts have different columns and are not covered here.
"""

from __future__ import annotations

import traceback
from typing import Any

import pandas as pd

from automl_and_llm_benchmark.common import paths

RESULT_COLUMNS = [
    "duration_training",
    "duration_prediction",
    "automl_system",
    "dataset",
    "hold_out_f1_score",
    "validation_best_score",
    "validation_all_scores",
    "5_fold_validation_scores",
    "hold_out_y_true",
    "hold_out_y_pred",
    "hold_out_y_pred_proba",
    "name_best_classifier",
]
METHOD_COLUMNS = ["method", "percentage"]

PLACEHOLDER_SCORES = [0, 0, 0, 0, 0]
ERROR_CLASSIFIER_NAME = "Error"


def _cell(values: Any) -> list[Any]:
    """Wrap a sequence/array into a one-element list so the DataFrame cell holds the list."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [list(values)]


def _placeholder() -> list[list[int]]:
    return [list(PLACEHOLDER_SCORES)]


def _build(data: dict, method: str | None, percentage: Any | None) -> pd.DataFrame:
    if method is not None or percentage is not None:
        data["method"] = method
        data["percentage"] = percentage
    return pd.DataFrame(data)


def summary_row(
    automl_system: str,
    dataset: str,
    duration_training: Any,
    duration_prediction: Any,
    hold_out_f1_score: float,
    validation_best_score: float,
    hold_out_y_true: Any,
    hold_out_y_pred: Any,
    name_best_classifier: str,
    validation_all_scores: Any | None = None,
    method: str | None = None,
    percentage: Any | None = None,
) -> pd.DataFrame:
    """Build the one-row summary DataFrame of a successful run.

    ``5_fold_validation_scores`` and ``hold_out_y_pred_proba`` are the placeholder
    ``[[0, 0, 0, 0, 0]]``; ``validation_all_scores`` defaults to it too.
    ``method`` and ``percentage`` (the ``*_method`` variants) are appended as the last
    columns when given.
    """
    data = {
        "duration_training": duration_training,
        "duration_prediction": duration_prediction,
        "automl_system": automl_system,
        "dataset": dataset,
        "hold_out_f1_score": hold_out_f1_score,
        "validation_best_score": validation_best_score,
        "validation_all_scores": (
            _placeholder() if validation_all_scores is None else _cell(validation_all_scores)
        ),
        "5_fold_validation_scores": _placeholder(),
        "hold_out_y_true": _cell(hold_out_y_true),
        "hold_out_y_pred": _cell(hold_out_y_pred),
        "hold_out_y_pred_proba": _placeholder(),
        "name_best_classifier": [name_best_classifier],
    }
    return _build(data, method, percentage)


def error_row(
    automl_system: str,
    dataset: str,
    method: str | None = None,
    percentage: Any | None = None,
) -> pd.DataFrame:
    """Build the one-row summary DataFrame written when a run fails (all zeros)."""
    data = {
        "duration_training": 0,
        "duration_prediction": 0,
        "automl_system": automl_system,
        "dataset": dataset,
        "hold_out_f1_score": 0,
        "validation_best_score": 0,
        "validation_all_scores": _placeholder(),
        "5_fold_validation_scores": _placeholder(),
        "hold_out_y_true": _placeholder(),
        "hold_out_y_pred": _placeholder(),
        "hold_out_y_pred_proba": _placeholder(),
        "name_best_classifier": [ERROR_CLASSIFIER_NAME],
    }
    return _build(data, method, percentage)


def write_summary(df: pd.DataFrame, filename: str) -> None:
    """Write ``df`` as a ``;``-separated CSV (no index) to ``WORK_DIR/filename``."""
    df.to_csv(paths.output_path(filename), sep=";", index=False)


def write_error(filename: str, exc: BaseException) -> None:
    """Write ``Error: <exc>`` plus the traceback to ``WORK_DIR/filename``.

    Call it from inside the ``except`` block: the traceback comes from
    ``traceback.format_exc()``, which only has content while an exception is being handled.
    """
    with open(paths.output_path(filename), "w") as fh:
        fh.write(f"Error: {exc}\n{traceback.format_exc()}")
