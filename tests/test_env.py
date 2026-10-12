import pytest

from automl_and_llm_benchmark.common import env


def test_get_env_unset_returns_default(monkeypatch):
    monkeypatch.delenv("T5A_VAR", raising=False)
    assert env.get_env("T5A_VAR") is None
    assert env.get_env("T5A_VAR", "x") == "x"


def test_get_env_empty_counts_as_unset(monkeypatch):
    monkeypatch.setenv("T5A_VAR", "")
    assert env.get_env("T5A_VAR") is None
    assert env.get_env("T5A_VAR", "fallback") == "fallback"


def test_get_env_value(monkeypatch):
    monkeypatch.setenv("T5A_VAR", "abc")
    assert env.get_env("T5A_VAR", "x") == "abc"


@pytest.mark.parametrize("value", [None, ""])
def test_require_env_missing(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("T5A_VAR", raising=False)
    else:
        monkeypatch.setenv("T5A_VAR", value)
    with pytest.raises(env.MissingEnvVarError, match="T5A_VAR"):
        env.require_env("T5A_VAR")


def test_missing_error_is_runtime_error():
    assert issubclass(env.MissingEnvVarError, RuntimeError)


def test_require_env_value(monkeypatch):
    monkeypatch.setenv("T5A_VAR", "v")
    assert env.require_env("T5A_VAR") == "v"


def test_get_int_env(monkeypatch):
    monkeypatch.delenv("T5A_INT", raising=False)
    assert env.get_int_env("T5A_INT") is None
    assert env.get_int_env("T5A_INT", 7) == 7
    monkeypatch.setenv("T5A_INT", "")
    assert env.get_int_env("T5A_INT", 7) == 7
    monkeypatch.setenv("T5A_INT", "12")
    assert env.get_int_env("T5A_INT", 7) == 12


def test_get_int_env_invalid(monkeypatch):
    monkeypatch.setenv("T5A_INT", "abc")
    with pytest.raises(ValueError, match="T5A_INT"):
        env.get_int_env("T5A_INT")


def test_require_int_env(monkeypatch):
    monkeypatch.setenv("T5A_INT", "3")
    assert env.require_int_env("T5A_INT") == 3
    monkeypatch.delenv("T5A_INT")
    with pytest.raises(env.MissingEnvVarError):
        env.require_int_env("T5A_INT")
