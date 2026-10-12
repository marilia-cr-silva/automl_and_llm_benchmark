import numpy as np
import pandas as pd
import pytest

from automl_and_llm_benchmark.common import io


def test_csv_round_trip_relative_to_work_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    df = pd.DataFrame({"text": ["a;b", "c"], "label": ["x", "y"]})
    io.write_csv(df, "sub/out.csv")
    assert (tmp_path / "sub" / "out.csv").read_text().splitlines()[0] == "text;label"
    pd.testing.assert_frame_equal(io.read_csv("sub/out.csv"), df)
    pd.testing.assert_frame_equal(io.read_csv(tmp_path / "sub" / "out.csv"), df)


def test_write_csv_index_option(monkeypatch, tmp_path):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    io.write_csv(pd.DataFrame({"a": [1]}), "i.csv", index=True)
    assert (tmp_path / "i.csv").read_text().splitlines()[0] == ";a"


def test_embeddings_round_trip_brotli(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    df = pd.DataFrame(
        {"text": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "label": [1, 0]},
    )
    io.write_embeddings(df, "emb.parquet.brotli")
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(tmp_path / "emb.parquet.brotli").metadata
    assert meta.row_group(0).column(0).compression == "BROTLI"
    back = io.read_embeddings("emb.parquet.brotli")
    assert [list(v) for v in back["text"]] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert back["label"].tolist() == [1, 0]


def test_embeddings_to_arrays():
    df = pd.DataFrame({"text": [[1.0, 2.0], [3.0, 4.0]], "label": [1, 0]})
    X, y = io.embeddings_to_arrays(df)
    assert X.shape == (2, 2)
    assert y.tolist() == ["1", "0"]
    assert y.dtype.kind == "U"
    np.testing.assert_array_equal(X, np.array(df["text"].tolist()))


def test_embeddings_to_arrays_keep_label_type():
    df = pd.DataFrame({"text": [[1.0], [2.0]], "label": [1, 0]})
    _, y = io.embeddings_to_arrays(df, label_as_str=False)
    assert y.tolist() == [1, 0]
