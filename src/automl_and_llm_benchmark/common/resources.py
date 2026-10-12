"""Access to the static data shipped inside the package (``automl_and_llm_benchmark.static``)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

_STATIC_PACKAGE = "automl_and_llm_benchmark.static"


def static_path(*parts: str) -> Path:
    """Return the path of a file inside ``automl_and_llm_benchmark.static``.

    Example: ``static_path("input_metadata", "gmm_25.csv")``. The package is installed
    unzipped (hatchling wheels), so the result is a real filesystem path.
    """
    ref = resources.files(_STATIC_PACKAGE)
    for part in parts:
        ref = ref.joinpath(part)
    return Path(str(ref))


def read_static_csv(*parts: str, **kwargs: Any):
    """Read a ``;``-separated static CSV into a DataFrame."""
    import pandas as pd  # lazy: keeps this module free of numpy/pandas at import time

    kwargs.setdefault("sep", ";")
    return pd.read_csv(static_path(*parts), **kwargs)
