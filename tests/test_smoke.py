"""Smoke tests: the package and its subpackages import."""

import automl_and_llm_benchmark
import automl_and_llm_benchmark.common


def test_version_is_string():
    assert isinstance(automl_and_llm_benchmark.__version__, str)


def test_common_importable():
    assert automl_and_llm_benchmark.common.__doc__
