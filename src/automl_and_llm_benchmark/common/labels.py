"""Label normalisation and per-dataset label-renaming maps, per dataset collection.

The canonical collection (the ``domain_adaptation`` datasets) is the default; the adjacent
``pt`` collection is selected with ``collection="pt"``. See
``automl_and_llm_benchmark.common.collection``.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import pandas as pd

from automl_and_llm_benchmark.common.collection import normalize_collection, static_parts
from automl_and_llm_benchmark.common.resources import static_path


def normalize_labels(series: pd.Series) -> pd.Series:
    """Lower-case a label column.

    Boolean columns become the strings ``"true"``/``"false"``. Other columns are
    lower-cased element-wise; if any element is not a string (e.g. integer labels) the
    column is returned unchanged.
    """
    if pd.api.types.is_bool_dtype(series):
        series = series.apply(lambda x: str(x).lower())
    try:
        series = series.apply(lambda x: x.lower())
    except Exception:  # noqa: BLE001, S110 - intentional: keep non-string columns unchanged
        pass
    return series


@functools.cache
def _load_collection(collection: str | None) -> dict[str, list[Any]]:
    with open(static_path(*static_parts(collection, "label_maps.json")), encoding="utf-8") as fh:
        return json.load(fh)


def available_datasets(collection: str | None = None) -> list[str]:
    """Return the dataset ids that have a label map in ``collection`` (``None`` = canonical)."""
    return list(_load_collection(normalize_collection(collection)))


def load_label_map(dataset: str, collection: str | None = None) -> dict[Any, Any]:
    """Return the label-renaming dict for ``dataset`` in ``collection`` (a fresh copy).

    ``collection`` is ``None``/``""`` for the canonical datasets (the default, used by the
    04/05/06 domain-adaptation stages) or ``"pt"`` for the adjacent Portuguese datasets
    (07/08). The two collections reuse the ids ed_01, fn_01, hs_01, pc_01 and sd_01..sd_04
    with DIFFERENT label meanings, so an id is only meaningful with its collection; never
    mix them. Key types (int vs str) and order are kept exactly as stored in the JSON.
    """
    collection = normalize_collection(collection)
    data = _load_collection(collection)
    if dataset not in data:
        raise KeyError(
            f"Unknown dataset {dataset!r} in label-map collection "
            f"{collection or 'canonical'!r}; available: {list(data)}"
        )
    return {key: value for key, value in data[dataset]}


def rename_labels(series: pd.Series, mapping: dict[Any, Any]) -> pd.Series:
    """Rename labels with ``series.replace(mapping)``."""
    return series.replace(mapping)
