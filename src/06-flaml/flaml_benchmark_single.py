# %%
"""FLAML AutoML baseline on one dataset: train on the full train split, predict the test split.

Environment variables:
    DATASET    - dataset id, used in the output file names (required)
    TEST_FILE  - test embeddings parquet file name under WORK_DIR (required)
    TRAIN_FILE - train embeddings parquet file name under WORK_DIR (required)
    WORK_DIR   - directory with the inputs and receiving the outputs (default /data)
    COLLECTION - "pt" reads and writes under WORK_DIR/pt (default: unset = WORK_DIR)
Inputs (produced by stage 01-embedding_generation):
    WORK_DIR/<TEST_FILE>, WORK_DIR/<TRAIN_FILE>
Outputs:
    WORK_DIR/flaml_summary_<DATASET>.csv
    WORK_DIR/flaml_automl_<DATASET>.log (FLAML search log)
    WORK_DIR/flaml_error_<DATASET>.txt (only when the run fails)
"""
from flaml.automl.data import *
from flaml.automl.data import get_output_from_log
from flaml import AutoML
from sklearn.metrics import f1_score
import datetime
from automl_and_llm_benchmark.common import io, results
from automl_and_llm_benchmark.common.env import require_env
from automl_and_llm_benchmark.common.paths import output_path

# %%
dataset = require_env("DATASET")
test_file = require_env("TEST_FILE")
train_file = require_env("TRAIN_FILE")

print(f"{dataset=} {test_file=} {train_file=}")

AUTOML_SYSTEM = "FLAML"
summary_filename = f"flaml_summary_{dataset}.csv"

try:
    df_test = io.read_embeddings(test_file)
    df_train = io.read_embeddings(train_file)

    X_train, y_train = io.embeddings_to_arrays(df_train)
    X_test, y_true = io.embeddings_to_arrays(df_test)
    time_stamps = {}
    time_stamps["beginning (training)"] = datetime.datetime.today()

    flaml_automl = AutoML()
    flaml_automl_settings = {
        "task": "classification",
        "time_budget": 60 * 60 * 15,  # seconds
        "log_file_name": str(output_path(f"flaml_automl_{dataset}.log")),
        "metric": "micro_f1",
    }
    flaml_automl.fit(X_train=X_train, y_train=y_train, **flaml_automl_settings)
    time_stamps["end (training)"] = datetime.datetime.today()

    duration_training = (
        time_stamps["end (training)"] - time_stamps["beginning (training)"]
    )

    name_estimator = flaml_automl.best_estimator
    dict_best_config = flaml_automl.best_config
    name_best_classifier = name_estimator + " " + str(dict_best_config)
    best_validation_score = flaml_automl.best_loss  # f1
    (
        time_history,
        best_valid_loss_history,
        valid_loss_history,
        config_history,
        metric_history,
    ) = get_output_from_log(
        filename=flaml_automl_settings["log_file_name"], time_budget=60000
    )
    all_validation_metrics = best_valid_loss_history

    time_stamps["beginning (prediction)"] = datetime.datetime.today()
    y_pred = flaml_automl.predict(X_test)
    time_stamps["end (prediction)"] = datetime.datetime.today()

    f1_score_hold_out = f1_score(y_true, y_pred, average="micro")

    duration_prediction = (
        time_stamps["end (prediction)"] - time_stamps["beginning (prediction)"]
    )

    df_dataset_summary = results.summary_row(
        automl_system=AUTOML_SYSTEM,
        dataset=dataset,
        duration_training=duration_training,
        duration_prediction=duration_prediction,
        hold_out_f1_score=f1_score_hold_out,
        validation_best_score=best_validation_score,  # F1-Score
        validation_all_scores=all_validation_metrics,  # F1
        hold_out_y_true=y_true,
        hold_out_y_pred=y_pred,
        name_best_classifier=name_best_classifier,
    )
    results.write_summary(df_dataset_summary, summary_filename)
except Exception as e:
    results.write_summary(results.error_row(AUTOML_SYSTEM, dataset), summary_filename)
    results.write_error(f"flaml_error_{dataset}.txt", e)
