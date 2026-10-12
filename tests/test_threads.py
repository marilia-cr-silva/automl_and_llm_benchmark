import os

import pytest

from automl_and_llm_benchmark.common import threads

VARS = ["NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # monkeypatch restores the previous values of everything it touched at teardown
    for name in VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_written_as_str():
    assert threads.configure_threads() == 16
    for name in threads.THREAD_VARIABLES:
        assert os.environ[name] == "16"


def test_default_param_and_num_threads(monkeypatch):
    assert threads.configure_threads(default=4) == 4
    assert os.environ["MKL_NUM_THREADS"] == "4"
    for name in threads.THREAD_VARIABLES:
        monkeypatch.delenv(name)
    monkeypatch.setenv("NUM_THREADS", "8")
    assert threads.configure_threads(default=4) == 8
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"


def test_existing_variable_wins(monkeypatch):
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    monkeypatch.setenv("NUM_THREADS", "8")
    threads.configure_threads()
    assert os.environ["MKL_NUM_THREADS"] == "3"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_empty_variable_counts_as_unset(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "")
    threads.configure_threads(default=5)
    assert os.environ["OMP_NUM_THREADS"] == "5"


def test_omp_threads_override_only_omp():
    assert threads.configure_threads(default=16, omp_threads=1) == 16
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "16"
    assert os.environ["MKL_NUM_THREADS"] == "16"


def test_explicit_omp_env_beats_omp_threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    threads.configure_threads(omp_threads=1)
    assert os.environ["OMP_NUM_THREADS"] == "2"
