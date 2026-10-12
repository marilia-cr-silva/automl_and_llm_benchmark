# %%
"""
This .py file runs the AutoGluon AutoML System based on a single dataset

Environment variables:
    DATASET      - name of the target dataset, e.g. "hs_01" (required)
    TEST_FILE    - the dataset's test set embeddings file name under WORK_DIR (required)
    TRAIN_FILE   - the dataset's train set embeddings file name under WORK_DIR (required)
    WORK_DIR     - reference directory, all supporting models and results are saved
                   relatively to it (default /data)
    COLLECTION   - "pt" reads and writes under WORK_DIR/pt (default: unset = WORK_DIR)
    NUM_THREADS  - BLAS/OpenMP threads (default 32)

Inputs:  <TEST_FILE> and <TRAIN_FILE> (embeddings parquet files from stage 01)
Outputs: WORK_DIR/autogluon_summary_{DATASET}.csv (final result)
         WORK_DIR/autogluon_{DATASET}/ (AutoGluon model directory, replaced if it exists)
         WORK_DIR/autogluon_error_{DATASET} (stack trace, only on error; note: no ".txt")

@article{agtabular,
  title={AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data},
  author={Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2003.06505},
  year={2020}
}
"""
# %%
from automl_and_llm_benchmark.common.threads import configure_threads

configure_threads(32)

import os
import warnings

warnings.filterwarnings("ignore")

from automl_and_llm_benchmark.common import io, paths, results
from automl_and_llm_benchmark.common.env import require_env

dataset = require_env("DATASET")
test_file = require_env("TEST_FILE")
train_file = require_env("TRAIN_FILE")

print(f"{dataset=} {test_file=} {train_file=}")

# %%
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from autogluon.tabular import TabularPredictor
import shutil

try:
    model_dir = str(paths.output_path(f"autogluon_{dataset}")) + os.sep

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    os.mkdir(model_dir)

    df_test = io.read_embeddings(test_file)
    df_train = io.read_embeddings(train_file)

    X_train_embeddings, y_train_labels = io.embeddings_to_arrays(df_train)
    X_train = pd.DataFrame(X_train_embeddings)
    y_train = pd.DataFrame(y_train_labels)
    y_train.columns = ["label"]
    X_train = pd.concat([X_train, y_train], axis=1)
    X_test_embeddings, y_true = io.embeddings_to_arrays(df_test)
    X_test = pd.DataFrame(X_test_embeddings)

    number_unique_label = len(np.unique(y_train_labels))

    if number_unique_label == 2:
        type_of_problem = "binary"
    else:
        type_of_problem = "multiclass"

    time_stamps = {}
    time_stamps["beginning (training)"] = datetime.datetime.today()
    predictor = TabularPredictor(
        label="label",
        problem_type=type_of_problem,
        eval_metric="f1_micro",
        path=model_dir,
    )
    predictor.fit(
        X_train,
        time_limit=60 * 60 * 24,
        presets="high_quality",
        num_gpus=1,
        ag_args_ensemble={"num_folds_parallel": 2},
    )
    time_stamps["end (training)"] = datetime.datetime.today()
    time_stamps["beginning (prediction)"] = datetime.datetime.today()
    y_pred = predictor.predict(X_test)
    time_stamps["end (prediction)"] = datetime.datetime.today()
    f1_score_hold_out = f1_score(y_true, y_pred, average="micro")

    summary_autogluon = predictor.fit_summary()
    ranked_models = summary_autogluon["leaderboard"]
    best_validation_score = ranked_models.iloc[0]["score_val"]
    all_validation_metrics = ranked_models["score_val"].dropna().tolist()
    best_model_name = ranked_models.iloc[0]["model"]
    name_best_classifier = (
        best_model_name + " " + summary_autogluon["model_types"][best_model_name]
    )

    duration_training = (
        time_stamps["end (training)"] - time_stamps["beginning (training)"]
    )
    duration_prediction = (
        time_stamps["end (prediction)"] - time_stamps["beginning (prediction)"]
    )

    df_dataset_summary = results.summary_row(
        automl_system="AutoGluon",
        dataset=dataset,
        duration_training=duration_training,
        duration_prediction=duration_prediction,
        hold_out_f1_score=f1_score_hold_out,
        validation_best_score=best_validation_score,
        validation_all_scores=all_validation_metrics,
        hold_out_y_true=y_true,
        hold_out_y_pred=y_pred,
        name_best_classifier=name_best_classifier,
    )
    results.write_summary(df_dataset_summary, f"autogluon_summary_{dataset}.csv")
except Exception as e:
    print(
        f"item 0: {test_file} and item 1: {train_file}"
    )
    df_dataset_summary = results.error_row("AutoGluon", dataset)
    results.write_summary(df_dataset_summary, f"autogluon_summary_{dataset}.csv")
    # the error file name has no ".txt" extension
    results.write_error(f"autogluon_error_{dataset}", e)
