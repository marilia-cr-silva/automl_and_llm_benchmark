# %%
"""FLAML AutoML baseline on the GMM-selected training subsets.

For every row of the stage-02 selected-instances CSV, trains on the selected train
instances and predicts the test split.

Environment variables:
    DATASET  - dataset id, used in the output file names (required)
    FILENAME - selected-instances CSV name under WORK_DIR (required)
    WORK_DIR - directory with the inputs and receiving the outputs (default /data)
    COLLECTION - "pt" reads and writes under WORK_DIR/pt (default: unset = WORK_DIR)
Inputs:
    WORK_DIR/<FILENAME> (stage 02-gmm; columns test;train;selected_train_instance_idx;
    method;percentage;dataset) and the embeddings parquets named in its test/train columns
    (stage 01-embedding_generation), resolved under WORK_DIR
Outputs (one per CSV row):
    WORK_DIR/flaml_summary_<method>_<DATASET>_<percentage>.csv
    WORK_DIR/flaml_error_<method>_<DATASET>_<percentage>.txt (only on failure)
    WORK_DIR/flaml_automl_3_<DATASET>_4.log (FLAML search log, see the note in the loop)
"""
import numpy as np
from sklearn.metrics import f1_score
from flaml.automl.data import *
from flaml.automl.data import get_output_from_log
from flaml import AutoML
import datetime
from automl_and_llm_benchmark.common import io, results
from automl_and_llm_benchmark.common.env import require_env
from automl_and_llm_benchmark.common.paths import output_path
from automl_and_llm_benchmark.common.selection import select_instances

# %%
dataset = require_env("DATASET")
filename = require_env("FILENAME")

AUTOML_SYSTEM = "FLAML"

TEST_FILE = 0
TRAIN_FILE = 1
SELECT_TRAIN_INSTANCE_IDX = 2
METHOD = 3
PERCENTAGE = 4

df_parquet_files = io.read_csv(filename)
array_parquet_files = np.array(df_parquet_files)
number_pairs, _ = array_parquet_files.shape

for counter in range(number_pairs):
    method = array_parquet_files[counter][METHOD]
    percentage = array_parquet_files[counter][PERCENTAGE]
    test_file = array_parquet_files[counter][TEST_FILE]
    train_file = array_parquet_files[counter][TRAIN_FILE]
    select_train_instance_idx = array_parquet_files[counter][SELECT_TRAIN_INSTANCE_IDX]
    summary_filename = f"flaml_summary_{method}_{dataset}_{percentage}.csv"

    try:
        df_test = io.read_embeddings(test_file)
        df_train = io.read_embeddings(train_file)

        # Filter records in train dataset
        df_train = select_instances(df_train, select_train_instance_idx)

        X_train, y_train = io.embeddings_to_arrays(df_train)
        X_test, y_true = io.embeddings_to_arrays(df_test)

        time_stamps = {}
        time_stamps["beginning (training)"] = datetime.datetime.today()

        flaml_automl = AutoML()
        flaml_automl_settings = {
            "task": "classification",
            "time_budget": 60 * 60 * 24,  # seconds
            # NOTE: METHOD and PERCENTAGE are the integer column positions (3 and 4), not
            # the row values, so every row logs to the same file ("flaml_automl_3_<dataset>_4.log").
            "log_file_name": str(
                output_path(f"flaml_automl_{METHOD}_{dataset}_{PERCENTAGE}.log")
            ),
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
            method=method,
            percentage=percentage,
        )
        results.write_summary(df_dataset_summary, summary_filename)
    except Exception as e:
        print(
            f"Error: {test_file=} {train_file=} {method=} {percentage=}"
        )
        # The error summary has no method/percentage columns.
        df_dataset_summary = results.error_row(AUTOML_SYSTEM, dataset)
        results.write_summary(df_dataset_summary, summary_filename)
        results.write_error(f"flaml_error_{method}_{dataset}_{percentage}.txt", e)
