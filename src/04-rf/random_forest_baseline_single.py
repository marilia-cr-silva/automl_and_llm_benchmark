# %%
"""Random Forest (random search) baseline on one dataset: train on the full train split.

Environment variables:
    DATASET    - dataset id, used in the output file names (required)
    TEST_FILE  - test embeddings parquet file name under WORK_DIR (required)
    TRAIN_FILE - train embeddings parquet file name under WORK_DIR (required)
    WORK_DIR   - directory with the inputs and receiving the outputs (default /data)
    COLLECTION - "pt" reads and writes under WORK_DIR/pt (default: unset = WORK_DIR)
Inputs (produced by stage 01-embedding_generation):
    WORK_DIR/<TEST_FILE>, WORK_DIR/<TRAIN_FILE>
Outputs:
    WORK_DIR/random_forest_summary_<DATASET>.csv
    WORK_DIR/random_forest_error_<DATASET>.txt (only when the run fails)
"""
import warnings

warnings.filterwarnings("ignore")
import datetime
from sklearn.metrics import f1_score
from automl_and_llm_benchmark.common import io, results
from automl_and_llm_benchmark.common.env import require_env
from automl_and_llm_benchmark.common.random_forest import build_search

# %%
dataset = require_env("DATASET")
test_file = require_env("TEST_FILE")
train_file = require_env("TRAIN_FILE")

print(f"{dataset=} {test_file=} {train_file=}")

AUTOML_SYSTEM = "Random Forest with Random Search"
summary_filename = f"random_forest_summary_{dataset}.csv"

try:
    df_test = io.read_embeddings(test_file)
    df_train = io.read_embeddings(train_file)

    X_train, y_train = io.embeddings_to_arrays(df_train)
    X_test, y_true = io.embeddings_to_arrays(df_test)

    time_stamps = {}
    time_stamps["beginning (training)"] = datetime.datetime.today()
    clf = build_search()
    random_search_clf = clf.fit(X_train, y_train)
    time_stamps["end (training)"] = datetime.datetime.today()
    time_stamps["beginning (prediction)"] = datetime.datetime.today()
    y_pred = random_search_clf.predict(X_test)
    time_stamps["end (prediction)"] = datetime.datetime.today()

    f1_score_hold_out = f1_score(y_true, y_pred, average="micro")

    duration_training = (
        time_stamps["end (training)"] - time_stamps["beginning (training)"]
    )
    duration_prediction = (
        time_stamps["end (prediction)"] - time_stamps["beginning (prediction)"]
    )
    df_dataset_summary = results.summary_row(
        automl_system=AUTOML_SYSTEM,
        dataset=dataset,
        duration_training=duration_training,
        duration_prediction=duration_prediction,
        hold_out_f1_score=f1_score_hold_out,
        validation_best_score=clf.best_score_,
        hold_out_y_true=y_true,
        hold_out_y_pred=y_pred,
        name_best_classifier="Random Forest",
    )
    results.write_summary(df_dataset_summary, summary_filename)
except Exception as e:
    results.write_summary(results.error_row(AUTOML_SYSTEM, dataset), summary_filename)
    results.write_error(f"random_forest_error_{dataset}.txt", e)

# %%
