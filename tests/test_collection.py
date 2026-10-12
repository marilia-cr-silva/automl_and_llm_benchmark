import pytest

from automl_and_llm_benchmark.common import collection


def test_unset_and_empty_are_canonical(monkeypatch):
    assert collection.current_collection() is None
    monkeypatch.setenv("COLLECTION", "")
    assert collection.current_collection() is None


def test_pt_is_selected(monkeypatch):
    monkeypatch.setenv("COLLECTION", "pt")
    assert collection.current_collection() == "pt"


def test_unknown_collection_raises_and_lists_valid_values(monkeypatch):
    monkeypatch.setenv("COLLECTION", "domain_adaptation")
    with pytest.raises(ValueError, match="domain_adaptation") as info:
        collection.current_collection()
    assert "pt" in str(info.value)


def test_normalize_collection():
    assert collection.normalize_collection(None) is None
    assert collection.normalize_collection("") is None
    assert collection.normalize_collection("pt") == "pt"
    with pytest.raises(ValueError):
        collection.normalize_collection("PT")


def test_require_canonical_passes_when_unset(monkeypatch):
    collection.require_canonical("04 test stage")
    monkeypatch.setenv("COLLECTION", "")
    collection.require_canonical("04 test stage")


def test_require_canonical_raises_when_set(monkeypatch):
    monkeypatch.setenv("COLLECTION", "pt")
    with pytest.raises(RuntimeError, match="04 test stage") as info:
        collection.require_canonical("04 test stage")
    assert "canonical" in str(info.value)
    assert "COLLECTION" in str(info.value)


def test_static_parts():
    assert collection.static_parts(None, "label_maps.json") == ("label_maps.json",)
    assert collection.static_parts("", "input_metadata", "gmm_25.csv") == (
        "input_metadata",
        "gmm_25.csv",
    )
    assert collection.static_parts("pt", "label_maps.json") == ("pt", "label_maps.json")
    with pytest.raises(ValueError):
        collection.static_parts("xx", "label_maps.json")
