"""Dataset collections: the canonical one (default) and adjacent ones (``COLLECTION``).

The canonical collection is the ``domain_adaptation`` datasets (static data at the root of
``automl_and_llm_benchmark.static``). The Portuguese ``pt`` datasets are an *adjacent*
collection selected with ``COLLECTION=pt``; its static data lives under
``automl_and_llm_benchmark.static.pt`` and its runtime files under ``WORK_DIR/pt``. The two
collections reuse some dataset ids (``ed_01``, ``pc_01``, ...) for different datasets, so an id
is only meaningful together with its collection.

Unset or empty ``COLLECTION`` means canonical. Any other unknown value is an error.
"""

from __future__ import annotations

from automl_and_llm_benchmark.common.env import get_env

COLLECTION_VARIABLE = "COLLECTION"
ADJACENT_COLLECTIONS = ("pt",)


def normalize_collection(value: str | None) -> str | None:
    """Return ``None`` (canonical) for ``None``/``""``, the name for a known adjacent one.

    Raises ``ValueError`` listing the valid values for anything else.
    """
    if value is None or value == "":
        return None
    if value in ADJACENT_COLLECTIONS:
        return value
    raise ValueError(
        f"Unknown collection {value!r}; valid values: unset/empty (canonical) or one of "
        f"{list(ADJACENT_COLLECTIONS)}."
    )


def current_collection() -> str | None:
    """Return the collection selected by ``COLLECTION`` (``None`` = canonical)."""
    return normalize_collection(get_env(COLLECTION_VARIABLE))


def require_canonical(stage: str) -> None:
    """Raise unless ``COLLECTION`` is unset/empty, for stages that only support canonical data."""
    collection = current_collection()
    if collection is not None:
        raise RuntimeError(
            f"{stage} only supports the canonical datasets, but {COLLECTION_VARIABLE}="
            f"{collection!r} is set. Unset {COLLECTION_VARIABLE} (or leave it empty) to run it."
        )


def static_parts(collection: str | None, *parts: str) -> tuple[str, ...]:
    """Return the static path parts of ``parts`` inside ``collection``."""
    collection = normalize_collection(collection)
    if collection is None:
        return tuple(parts)
    return (collection, *parts)
