import numpy as np
import pandas as pd

from automl_and_llm_benchmark.common import results

# Header of `random_forest_summary_<dataset>.csv`; the `*_method` variants append
# `method;percentage`.
EXPECTED_SUMMARY_HEADER = (
    "duration_training;duration_prediction;automl_system;dataset;hold_out_f1_score;"
    "validation_best_score;validation_all_scores;5_fold_validation_scores;hold_out_y_true;"
    "hold_out_y_pred;hold_out_y_pred_proba;name_best_classifier"
)


def make_row(**kwargs):
    return results.summary_row(
        automl_system="Random Forest with Random Search",
        dataset="fx_01",
        duration_training=pd.Timedelta(seconds=1),
        duration_prediction=pd.Timedelta(milliseconds=16),
        hold_out_f1_score=0.8,
        validation_best_score=0.9,
        hold_out_y_true=np.array(["pos", "neg", "pos"]),
        hold_out_y_pred=["pos", "pos", "neg"],
        name_best_classifier="Random Forest",
        **kwargs,
    )


def test_result_columns_order():
    assert results.RESULT_COLUMNS == [
        "duration_training",
        "duration_prediction",
        "automl_system",
        "dataset",
        "hold_out_f1_score",
        "validation_best_score",
        "validation_all_scores",
        "5_fold_validation_scores",
        "hold_out_y_true",
        "hold_out_y_pred",
        "hold_out_y_pred_proba",
        "name_best_classifier",
    ]
    assert results.PLACEHOLDER_SCORES == [0, 0, 0, 0, 0]


def test_summary_row_values():
    df = make_row()
    assert list(df.columns) == results.RESULT_COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert row["hold_out_y_true"] == ["pos", "neg", "pos"]
    assert row["hold_out_y_pred"] == ["pos", "pos", "neg"]
    assert row["5_fold_validation_scores"] == [0, 0, 0, 0, 0]
    assert row["hold_out_y_pred_proba"] == [0, 0, 0, 0, 0]
    assert row["validation_all_scores"] == [0, 0, 0, 0, 0]
    assert row["name_best_classifier"] == "Random Forest"


def test_summary_row_validation_all_scores():
    df = make_row(validation_all_scores=[0.5, 0.6])
    assert df.iloc[0]["validation_all_scores"] == [0.5, 0.6]


def test_summary_row_method_columns_last():
    df = make_row(method="gmm_random", percentage=25)
    assert list(df.columns) == results.RESULT_COLUMNS + ["method", "percentage"]
    assert df.iloc[0]["method"] == "gmm_random"
    assert df.iloc[0]["percentage"] == 25


def test_placeholders_not_shared():
    first, second = make_row(), make_row()
    first.iloc[0]["5_fold_validation_scores"].append(1)
    assert second.iloc[0]["5_fold_validation_scores"] == [0, 0, 0, 0, 0]
    assert results.PLACEHOLDER_SCORES == [0, 0, 0, 0, 0]


def test_error_row():
    df = results.error_row("FLAML", "fx_01")
    assert list(df.columns) == results.RESULT_COLUMNS
    row = df.iloc[0]
    assert row["duration_training"] == 0
    assert row["duration_prediction"] == 0
    assert row["automl_system"] == "FLAML"
    assert row["dataset"] == "fx_01"
    assert row["hold_out_f1_score"] == 0
    assert row["validation_best_score"] == 0
    for col in [
        "validation_all_scores",
        "5_fold_validation_scores",
        "hold_out_y_true",
        "hold_out_y_pred",
        "hold_out_y_pred_proba",
    ]:
        assert row[col] == [0, 0, 0, 0, 0]
    assert row["name_best_classifier"] == "Error"


def test_error_row_method():
    df = results.error_row("FLAML", "fx_01", method="m", percentage=50)
    assert list(df.columns) == results.RESULT_COLUMNS + ["method", "percentage"]


def test_write_summary_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    results.write_summary(make_row(), "summary.csv")
    text = (tmp_path / "summary.csv").read_text()
    assert text.splitlines()[0] == ";".join(results.RESULT_COLUMNS)
    back = pd.read_csv(tmp_path / "summary.csv", sep=";")
    assert list(back.columns) == results.RESULT_COLUMNS
    assert back.iloc[0]["hold_out_y_true"] == "['pos', 'neg', 'pos']"
    assert back.iloc[0]["5_fold_validation_scores"] == "[0, 0, 0, 0, 0]"


def test_write_error(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    try:
        raise ValueError("boom")
    except ValueError as exc:
        results.write_error("err.txt", exc)
    content = (tmp_path / "err.txt").read_text()
    assert content.startswith("Error: boom\n")
    assert "Traceback" in content
    assert "ValueError: boom" in content


def test_written_header_format(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    results.write_summary(make_row(), "single.csv")
    results.write_summary(make_row(method="gmm_random", percentage=25), "method.csv")
    single_header = (tmp_path / "single.csv").read_text().splitlines()[0]
    method_header = (tmp_path / "method.csv").read_text().splitlines()[0]
    assert single_header == EXPECTED_SUMMARY_HEADER
    assert method_header == EXPECTED_SUMMARY_HEADER + ";method;percentage"
