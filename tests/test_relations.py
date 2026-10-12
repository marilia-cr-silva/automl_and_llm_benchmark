import pytest

from automl_and_llm_benchmark.common import relations


@pytest.fixture(scope="module")
def correspondence():
    return relations.get_dict_correspondence()


def test_correspondence_contains_related(correspondence):
    assert "ed_01" in correspondence["ed_03"]
    assert "ed_02" in correspondence["ed_03"]


def test_correspondence_values_sorted_unique(correspondence):
    for related in correspondence.values():
        assert related == sorted(set(related))


def test_training_and_prediction_lists():
    to_train, to_predict = relations.get_list_training_and_prediction("ed_03")
    assert "ed_03" not in to_train
    assert {"ed_01", "ed_02"} <= set(to_train)
    assert to_predict == to_train + ["ed_03"]


def test_unknown_dataset_raises_key_error():
    with pytest.raises(KeyError):
        relations.get_list_training_and_prediction("zz_99")


@pytest.fixture(scope="module")
def metadata():
    return relations.get_input_metadata()


def test_input_metadata_columns(metadata):
    expected = [
        "test",
        "train",
        "selected_train_instance_idx",
        "method",
        "percentage",
        "dataset",
    ]
    assert set(expected) <= set(metadata.columns)
    assert list(metadata.columns)[:6] == expected


def test_input_metadata_percentages_and_index(metadata):
    assert set(metadata["percentage"].unique()) == {25, 50, 75}
    assert metadata.index.tolist() == list(range(len(metadata)))
    # concatenation order: 25, then 50, then 75
    firsts = metadata["percentage"].drop_duplicates().tolist()
    assert firsts == [25, 50, 75]
