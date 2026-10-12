from pathlib import Path

import pytest

from automl_and_llm_benchmark.common import paths


def test_default_work_dir(monkeypatch):
    monkeypatch.delenv("WORK_DIR", raising=False)
    assert paths.work_dir() == Path("/data")


def test_empty_work_dir_uses_default(monkeypatch):
    monkeypatch.setenv("WORK_DIR", "")
    assert paths.work_dir() == Path("/data")


def test_work_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    assert paths.work_dir() == tmp_path


def test_resolve(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    assert paths.resolve("a/b.csv") == tmp_path / "a" / "b.csv"
    absolute = tmp_path / "elsewhere" / "x.csv"
    assert paths.resolve(absolute) == absolute
    assert paths.resolve(str(absolute)) == absolute


def test_output_path_creates_work_dir(monkeypatch, tmp_path):
    target = tmp_path / "new" / "dir"
    monkeypatch.setenv("WORK_DIR", str(target))
    out = paths.output_path("out.csv")
    assert out == target / "out.csv"
    assert target.is_dir()


def test_canonical_work_dir_ignores_collection_subdir(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("COLLECTION", "")
    assert paths.work_dir() == tmp_path
    assert paths.base_work_dir() == tmp_path


def test_pt_collection_moves_work_dir_but_not_base(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("COLLECTION", "pt")
    assert paths.work_dir() == tmp_path / "pt"
    assert paths.base_work_dir() == tmp_path
    assert paths.resolve("a/b.csv") == tmp_path / "pt" / "a" / "b.csv"
    absolute = tmp_path / "elsewhere" / "x.csv"
    assert paths.resolve(absolute) == absolute


def test_pt_collection_default_work_dir(monkeypatch):
    monkeypatch.delenv("WORK_DIR", raising=False)
    monkeypatch.setenv("COLLECTION", "pt")
    assert paths.work_dir() == Path("/data/pt")
    assert paths.base_work_dir() == Path("/data")


def test_output_path_under_collection_creates_subdir(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("COLLECTION", "pt")
    out = paths.output_path("out.csv")
    assert out == tmp_path / "pt" / "out.csv"
    assert (tmp_path / "pt").is_dir()


def test_unknown_collection_raises_in_work_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("COLLECTION", "xx")
    with pytest.raises(ValueError, match="xx"):
        paths.work_dir()
