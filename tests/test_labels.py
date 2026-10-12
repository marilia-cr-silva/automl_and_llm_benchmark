import pandas as pd
import pytest

from automl_and_llm_benchmark.common import labels


def test_available_datasets_counts():
    canonical = labels.available_datasets()
    pt = labels.available_datasets("pt")
    assert len(canonical) == 49
    assert len(pt) == 10
    assert len(set(canonical)) == 49
    assert len(set(pt)) == 10


def test_default_and_empty_collection_are_canonical():
    canonical = labels.available_datasets()
    assert labels.available_datasets("") == canonical
    assert labels.available_datasets(None) == canonical
    expected = labels.load_label_map("pc_01")
    assert labels.load_label_map("pc_01", "") == expected
    assert labels.load_label_map("pc_01", collection=None) == expected


def test_every_dataset_has_a_non_empty_label_map():
    for collection in (None, "pt"):
        for dataset in labels.available_datasets(collection):
            assert labels.load_label_map(dataset, collection), (collection, dataset)


def test_pt_hs_01_has_both_str_and_int_keys():
    mapping = labels.load_label_map("hs_01", "pt")
    assert list(mapping) == ["0", "1", 0, 1]


def test_canonical_pc_07_has_negative_key():
    mapping = labels.load_label_map("pc_07")
    assert -1 in mapping


def test_collections_differ_for_shared_ids():
    shared = ["ed_01", "fn_01", "hs_01", "pc_01", "sd_01", "sd_02", "sd_03", "sd_04"]
    for dataset in shared:
        assert labels.load_label_map(dataset, "pt") != labels.load_label_map(dataset)


def test_dataset_only_in_canonical_is_unknown_in_pt():
    assert "pc_07" not in labels.available_datasets("pt")
    with pytest.raises(KeyError, match="pc_07"):
        labels.load_label_map("pc_07", "pt")


def test_dataset_is_required():
    with pytest.raises(TypeError):
        labels.load_label_map()  # type: ignore[call-arg]


def test_load_returns_fresh_copy():
    first = labels.load_label_map("fn_01", "pt")
    first["extra"] = "x"
    assert "extra" not in labels.load_label_map("fn_01", "pt")


def test_unknown_dataset_names_available_ids():
    with pytest.raises(KeyError, match="ed_01"):
        labels.load_label_map("nope_99", "pt")


def test_unknown_collection():
    with pytest.raises(ValueError, match="pt"):
        labels.available_datasets("nope")
    with pytest.raises(ValueError, match="nope"):
        labels.load_label_map("ed_01", "nope")


def test_normalize_bool():
    out = labels.normalize_labels(pd.Series([True, False, True]))
    assert out.tolist() == ["true", "false", "true"]


def test_normalize_strings():
    out = labels.normalize_labels(pd.Series(["Fake", "TRUE", "ok"]))
    assert out.tolist() == ["fake", "true", "ok"]


def test_normalize_ints_unchanged():
    series = pd.Series([0, 1, 2])
    out = labels.normalize_labels(series)
    assert out.tolist() == [0, 1, 2]
    assert out.dtype == series.dtype


def test_rename_labels():
    series = pd.Series(["fake", "true", "fake"])
    out = labels.rename_labels(series, labels.load_label_map("fn_01", "pt"))
    assert out.tolist() == ["false", "true", "false"]
