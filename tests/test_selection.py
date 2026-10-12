import pandas as pd

from automl_and_llm_benchmark.common import selection


def test_parse_instance_indices():
    assert selection.parse_instance_indices("[0, 4, 5,  7]") == [0, 4, 5, 7]
    assert selection.parse_instance_indices("[3]") == [3]


def test_parse_empty_and_blank():
    assert selection.parse_instance_indices("[]") == []
    assert selection.parse_instance_indices("") == []
    assert selection.parse_instance_indices("   ") == []


def test_select_instances_from_raw_and_list():
    df = pd.DataFrame({"text": list("abcdef"), "label": range(6)})
    expected = ["a", "c", "f"]
    assert selection.select_instances(df, "[0, 2, 5]")["text"].tolist() == expected
    assert selection.select_instances(df, [0, 2, 5])["text"].tolist() == expected


def test_select_instances_resets_index():
    df = pd.DataFrame({"text": list("abcd")}, index=[10, 11, 12, 13])
    out = selection.select_instances(df, [1, 3])
    assert out["text"].tolist() == ["b", "d"]
    assert out.index.tolist() == [1, 3]


def test_select_instances_empty():
    df = pd.DataFrame({"text": list("abc")})
    assert selection.select_instances(df, "[]").empty
