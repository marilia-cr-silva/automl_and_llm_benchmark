"""Dataset relations and GMM input metadata used by the domain-adaptation stages.

Shared by the 04/05/06 ``domain_adaptation`` scripts; the CSVs are read from the packaged
static data.
"""

from __future__ import annotations

import pandas as pd

from automl_and_llm_benchmark.common.resources import read_static_csv

PERCENTAGES = (25, 50, 75)


def get_dict_correspondence() -> dict[str, list[str]]:
    """Map each ``first_dataset`` to the sorted datasets it contains (itself included).

    Uses ``str.contains`` query semantics, so an id that is a substring of
    another id also matches that one's rows.
    """
    dict_corresp: dict[str, list[str]] = {}
    df = read_static_csv("dataset_relations.csv")
    for unique_dataset in list(set(df["first_dataset"].tolist())):
        df_filtered = df.query(f"first_dataset.str.contains('{unique_dataset}')")
        dict_corresp[unique_dataset] = sorted(set(df_filtered["second_dataset"].tolist()))
    return dict_corresp


def get_list_training_and_prediction(dataset: str) -> tuple[list[str], list[str]]:
    """Return ``(to_train, to_predict)`` for the base ``dataset``.

    ``to_train`` are the related datasets without ``dataset`` itself; ``to_predict`` is
    ``to_train`` followed by ``dataset``.
    """
    related = get_dict_correspondence()[dataset]
    to_train = [item for item in related if item != dataset]
    to_predict = to_train.copy()
    to_predict.append(dataset)
    return to_train, to_predict


def get_input_metadata() -> pd.DataFrame:
    """Concatenate the GMM input metadata for 25%, 50% and 75% (index reset)."""
    frames = [read_static_csv("input_metadata", f"gmm_{pct}.csv") for pct in PERCENTAGES]
    return pd.concat(frames, axis=0).reset_index(drop=True)
