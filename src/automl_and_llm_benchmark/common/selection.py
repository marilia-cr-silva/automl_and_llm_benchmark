"""Selection of training instances from the stage-02 ``selected_train_instance_idx`` column."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def parse_instance_indices(raw: str) -> list[int]:
    """Parse a string such as ``"[0, 4, 5]"`` into ``[0, 4, 5]``.

    Strips the brackets, drops spaces and splits on commas. ``"[]"`` and blank strings give
    ``[]`` instead of raising.
    """
    inner = raw.strip()[1:-1].replace(" ", "")
    if inner == "":
        return []
    return [int(el) for el in inner.split(",")]


def select_instances(df: pd.DataFrame, indices_or_raw: str | Sequence[int]) -> pd.DataFrame:
    """Return the rows of ``df`` at the given positional indices.

    ``indices_or_raw`` is either the raw stage-02 string or a list of ints. The index is
    reset first, so indices are positions in ``df``. Callers that drop NaNs must do so
    before calling this.
    """
    if isinstance(indices_or_raw, str):
        indices = parse_instance_indices(indices_or_raw)
    else:
        indices = list(indices_or_raw)
    return df.reset_index(drop=True).loc[indices]
