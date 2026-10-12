"""Thread configuration for BLAS/OpenMP libraries.

Must be called BEFORE numpy/sklearn/torch are imported, so this module imports neither.
"""

from __future__ import annotations

import os

from automl_and_llm_benchmark.common.env import get_env, get_int_env

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def configure_threads(default: int = 16, omp_threads: int | None = None) -> int:
    """Set the BLAS/OpenMP thread environment variables and return the thread count.

    The count comes from ``NUM_THREADS`` (falling back to ``default``). Each of
    ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS`` and ``OMP_NUM_THREADS`` that is already
    set (and non-empty) keeps its value, so explicit per-variable overrides win. If
    ``omp_threads`` is given it replaces the count used for ``OMP_NUM_THREADS`` only.
    Values are written as ``str``.
    """
    number = get_int_env("NUM_THREADS", default)
    assert number is not None
    wanted: dict[str, int] = {name: number for name in THREAD_VARIABLES}
    if omp_threads is not None:
        wanted["OMP_NUM_THREADS"] = omp_threads
    for name, value in wanted.items():
        if get_env(name) is None:
            os.environ[name] = str(value)
    return number
