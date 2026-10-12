"""Working-directory handling: ``WORK_DIR`` (default ``/data``) plus the collection subdirectory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from automl_and_llm_benchmark.common.collection import current_collection

DEFAULT_WORK_DIR = "/data"

PathLike = Union[str, "os.PathLike[str]"]


def base_work_dir() -> Path:
    """Return the raw ``WORK_DIR`` (default ``/data``), the same for every collection."""
    value = os.environ.get("WORK_DIR")
    return Path(value if value else DEFAULT_WORK_DIR)


def work_dir() -> Path:
    """Return the I/O root: ``WORK_DIR`` (canonical) or ``WORK_DIR/<collection>`` (e.g. ``pt``)."""
    base = base_work_dir()
    collection = current_collection()
    return base if collection is None else base / collection


def resolve(path: PathLike) -> Path:
    """Return ``path`` unchanged if absolute, else resolved under :func:`work_dir`."""
    p = Path(path)
    if p.is_absolute():
        return p
    return work_dir() / p


def output_path(filename: PathLike) -> Path:
    """Return ``work_dir()/filename``, creating the working directory if missing."""
    directory = work_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename
