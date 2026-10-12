# %%
"""Random Forest (random search) baseline on the GMM-selected training subsets.

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
    WORK_DIR/random_forest_summary_<method>_<DATASET>_<percentage>.csv
    WORK_DIR/random_forest_error_<method>_<DATASET>_<percentage>.txt (only on failure)
"""
import warnings

warnings.filterwarnings("ignore")
import datetime
import numpy as np
from sklearn.metrics import f1_score
from automl_and_llm_benchmark.common import io, results
from automl_and_llm_benchmark.common.env import require_env
from automl_and_llm_benchmark.common.random_forest import build_search
from automl_and_llm_benchmark.common.selection import select_instances

TEST_FILE = 0
TRAIN_FILE = 1
SELECT_TRAIN_INSTANCE_IDX = 2
METHOD = 3
PERCENTAGE = 4

# %%
dataset = require_env("DATASET")
filename = require_env("FILENAME")

print(f"{filename=} {dataset=} ")

AUTOML_SYSTEM = "Random Forest with Random Search"

df_parquet_files = io.read_csv(filename)
array_parquet_files = np.array(df_parquet_files)
number_pairs, _ = array_parquet_files.shape

for counter in range(number_pairs):
    method = array_parquet_files[counter][METHOD]
    percentage = array_parquet_files[counter][PERCENTAGE]
    test_file = array_parquet_files[counter][TEST_FILE]
    train_file = array_parquet_files[counter][TRAIN_FILE]
    select_train_instance_idx = array_parquet_files[counter][SELECT_TRAIN_INSTANCE_IDX]
    summary_filename = f"random_forest_summary_{method}_{dataset}_{percentage}.csv"

    try:
        df_test = io.read_embeddings(test_file)
        df_train = io.read_embeddings(train_file)

        # Filter records in train dataset
        df_train = select_instances(df_train, select_train_instance_idx)

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
            method=method,
            percentage=percentage,
        )
        results.write_summary(df_dataset_summary, summary_filename)
    except Exception as e:
        print(
            f"Error: {test_file=} {train_file=} {method=} {percentage=}"
        )
        df_dataset_summary = results.error_row(
            AUTOML_SYSTEM, dataset, method=method, percentage=percentage
        )
        results.write_summary(df_dataset_summary, summary_filename)
        results.write_error(
            f"random_forest_error_{method}_{dataset}_{percentage}.txt", e
        )
