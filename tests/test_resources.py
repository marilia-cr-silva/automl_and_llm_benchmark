from automl_and_llm_benchmark.common import resources


def test_static_path_points_to_files():
    assert resources.static_path("dataset_relations.csv").is_file()
    assert resources.static_path("label_maps.json").is_file()
    assert resources.static_path("pt", "label_maps.json").is_file()
    assert resources.static_path("pt", "zero_and_few_shot_files.csv").is_file()
    assert resources.static_path("input_metadata", "gmm_25.csv").is_file()


def test_read_static_csv_uses_semicolon():
    df = resources.read_static_csv("dataset_relations.csv")
    assert list(df.columns) == ["first_dataset", "relation", "second_dataset"]


def test_read_static_csv_in_adjacent_collection():
    df = resources.read_static_csv("pt", "zero_and_few_shot_files.csv")
    assert list(df.columns) == ["name_dataset", "name_file_llm", "name_file_predict"]
