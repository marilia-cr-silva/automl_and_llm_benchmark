"""CSV and embeddings (parquet, brotli) I/O relative to ``WORK_DIR``.

Parquet access goes through ``pandas`` only, so pyarrow is imported lazily by pandas
(stages 07/08 have no pyarrow installed and never call the parquet functions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from automl_and_llm_benchmark.common import paths
from automl_and_llm_benchmark.common.paths import PathLike


def read_csv(path: PathLike, **kwargs: Any) -> pd.DataFrame:
    """Read a ``;``-separated CSV; relative paths are resolved under ``WORK_DIR``."""
    kwargs.setdefault("sep", ";")
    return pd.read_csv(paths.resolve(path), **kwargs)


def write_csv(df: pd.DataFrame, path: PathLike, index: bool = False, **kwargs: Any) -> None:
    """Write a ``;``-separated CSV; relative paths are resolved under ``WORK_DIR``."""
    kwargs.setdefault("sep", ";")
    target = paths.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=index, **kwargs)


def read_embeddings(path: PathLike) -> pd.DataFrame:
    """Read an embeddings parquet file (columns ``text`` = vector, ``label``)."""
    return pd.read_parquet(paths.resolve(path))


def write_embeddings(df: pd.DataFrame, path: PathLike) -> None:
    """Write an embeddings DataFrame as parquet with brotli compression."""
    target = paths.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, compression="brotli")


def embeddings_to_arrays(
    df: pd.DataFrame, label_as_str: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` arrays; the embedding vector lives in the ``text`` column.

    With ``label_as_str`` (the default) labels are converted to
    ``str`` before building ``y``.
    """
    labels = df["label"].astype(str) if label_as_str else df["label"]
    return np.array(df["text"].tolist()), np.array(labels.tolist())
