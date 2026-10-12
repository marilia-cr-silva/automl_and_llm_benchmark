# %%
"""FLAML AutoML baseline for domain adaptation by dataset concatenation, for ONE base dataset.

Trains on 100% of the "base" dataset's train split concatenated with the GMM-selected
(25/50/75%) train instances of each dataset related to it, and predicts the hold-out sets of
the related datasets and of the base dataset. Labels of every dataset are renamed to a common
vocabulary (label maps of the canonical collection).

Environment variables:
    DATASET    - base dataset id (e.g. ed_03) (required)
    WALL_TIME  - FLAML time budget per trained model, in hours, integer (required)
    WORK_DIR   - directory with the embeddings parquets and receiving the outputs (default /data)
    COLLECTION - must be unset/empty: this stage only supports the canonical datasets
Inputs (stage 01-embedding_generation, stage 02-gmm):
    embeddings parquets named in the static input_metadata/gmm_{25,50,75}.csv, under WORK_DIR
Outputs:
    WORK_DIR/part_flaml_100_combination_summary_gmm_random_<combination>.csv (one per combination)
    WORK_DIR/flaml_automl_base_<base>_concat_gmm_random_<dataset>_<percentage>.log (FLAML logs)
    WORK_DIR/flaml_error_100_combination_gmm_random_<DATASET>.txt (only when the run fails)
"""
import datetime
import traceback
from typing import Any, Union
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from flaml.automl.data import get_output_from_log
from flaml import AutoML
from automl_and_llm_benchmark.common import io, results
from automl_and_llm_benchmark.common.collection import require_canonical
from automl_and_llm_benchmark.common.env import require_env, require_int_env
from automl_and_llm_benchmark.common.labels import load_label_map, normalize_labels, rename_labels
from automl_and_llm_benchmark.common.paths import output_path
from automl_and_llm_benchmark.common.relations import get_input_metadata, get_list_training_and_prediction
from automl_and_llm_benchmark.common.selection import select_instances

require_canonical("06 FLAML domain adaptation")

# %%
SELECTED_DATASET = require_env("DATASET")
SELECTED_WALL_TIME = require_int_env("WALL_TIME")  # hours

# %%
def get_set_classifier_flaml(
    log_name_part_00: str,
    log_name_part_01: str,
    log_name_part_02: str,
    log_name_part_03: str,
) -> Union[Any]:
    """This function creates an estimator and returns
    the estimator and its settings specifying a path to the log file


    Args:
        log_name_part_00 (str): dataset responsible for being the base (100% is used)
        log_name_part_01 (str): method responsible for sampling the auxiliary dataset
        log_name_part_02 (str): dataset that is sampled (which is compatible with the base dataset)
        log_name_part_03 (str): used percentage of the sampled dataset

    Returns:
        Union[Any]: it returns an estimator and the settings of this estimator
    """
    flaml_automl_estimator = AutoML()
    flaml_automl_all_settings = {
        "task": "classification",
        "time_budget": 60 * 60 * SELECTED_WALL_TIME,  # seconds
        "log_file_name": str(
            output_path(
                f"flaml_automl_base_{log_name_part_00}_concat_{log_name_part_01}_{log_name_part_02}_{log_name_part_03}.log"
            )
        ),
        "metric": "micro_f1",
    }

    return flaml_automl_estimator, flaml_automl_all_settings


def get_training_concatenated_np_array(
    dataset_to_concat: str,
    selected_percentage: int,
    df_input: pd.DataFrame,
    df_base: pd.DataFrame,
) -> Union[np.array]:
    """This function retrieves the sampled datasets and filters by the selected dataset with its percentage
        that are going to be concatenated with the 'base' dataset.
        After extracting the indices from the dataframe, only these rows are selected to concatenate the 'base' dataset.
        All the labels are replaced based on the dictionary with their correspondence.
        The concatenated training set is split, as NumPy arrays, into embeddings and labels

    Args:
        dataset_to_concat (str): dataset that is sampled
        selected_percentage (int): the percentage of the data available in the sampled dataset
        df_input (pd.DataFrame): the dataframe that has the entire dataset that is going to be sampled
        df_base (pd.DataFrame): the 'base' dataset that is going to be used with 100% of its training set

    Returns:
        Union[np.array]: the concatenated ('base' dataset + a different sampled dataset) training set split into X (embeddings) and y (labels)
    """
    df_parquet_files_aux = df_input.query(
        f"dataset.str.contains('{dataset_to_concat}') and percentage=={selected_percentage}"
    )
    file_name_train = list(df_parquet_files_aux["train"])
    print(
        f"train: {file_name_train[0]} dataset_to_concat: {dataset_to_concat} percentage: {selected_percentage}"
    )
    df_sampled_to_concat = io.read_embeddings(file_name_train[0])
    # Filter records in train dataset
    select_train_instance_idx = list(
        df_parquet_files_aux["selected_train_instance_idx"]
    )[0]
    df_sampled_to_concat = select_instances(
        df_sampled_to_concat, select_train_instance_idx
    )
    dict_replacements = load_label_map(dataset_to_concat)
    df_sampled_to_concat["label"] = normalize_labels(df_sampled_to_concat["label"])
    df_sampled_to_concat["label"] = rename_labels(
        df_sampled_to_concat["label"], dict_replacements
    )
    df_concat = pd.concat([df_base, df_sampled_to_concat], axis=0).reset_index(
        drop=True
    )
    X_train_array = np.array(df_concat["text"].tolist())
    y_train_array = np.array(df_concat["label"].tolist())
    print("get_training_concat_np_array worked!")
    print(
        f"\n before: {df_base['label'].unique()} after: {df_concat['label'].unique()}"
    )
    return X_train_array, y_train_array


def get_numpy_array_prediction(
    prediction_dataset_selected: str, selected_percentage: int, df_input: pd.DataFrame
) -> Union[np.array]:
    """This function filters the metadata and returns two NumPy arrays with the hold-out embeddings and true labels

    Args:
        prediction_dataset_selected (str): dataset that is going to be predicted
        selected_percentage (int): percentage (only to reduce the number of rows to be searched)
        df_input (pd.DataFrame): dataframe with dataset metadata (such as file name, instance selection method, percentage)

    Returns:
        Union[np.array]: the hold-out set split into X (embeddings) and y (labels)
    """
    print(f"prediction_dataset: {prediction_dataset_selected}")
    df_parquet_files_aux = df_input.query(
        f"dataset.str.contains('{prediction_dataset_selected}') and percentage=={selected_percentage}"
    )
    file_name_test = list(df_parquet_files_aux["test"])
    df_test_aux = io.read_embeddings(file_name_test[0])
    dict_replacements = load_label_map(prediction_dataset_selected)
    df_test_aux["label"] = normalize_labels(df_test_aux["label"])
    df_test_aux["label"] = rename_labels(df_test_aux["label"], dict_replacements)

    X_test_array = np.array(df_test_aux["text"].tolist())
    y_true_array = np.array(df_test_aux["label"].tolist())
    print("get_numpy_array_prediction worked!")
    print(f"\n labels: {df_test_aux['label'].unique()}")
    return X_test_array, y_true_array


# %%
df_input_metadata = get_input_metadata()
df_parquet_files = df_input_metadata.query("dataset.str.contains(@SELECTED_DATASET)")
df_parquet_files = df_parquet_files.reset_index(drop=True)


try:
    task_and_file = SELECTED_DATASET
    # the 'base' is the dataset that uses 100% of its data and is combined with another sampled dataset
    df_train_base = io.read_embeddings(df_parquet_files["train"].iloc[0])
    dict_replacement = load_label_map(task_and_file)
    # the labels need to be lower case before replacement with corresponding new labels
    df_train_base["label"] = normalize_labels(df_train_base["label"])
    df_train_base["label"] = rename_labels(df_train_base["label"], dict_replacement)
    df_train_base = df_train_base.reset_index(drop=True)

    list_datasets_to_be_trained, list_datasets_to_be_predicted = (
        get_list_training_and_prediction(task_and_file)
    )

    # iterate over datasets that are going to be sampled and concatenated with the 'base' dataset
    for item in list_datasets_to_be_trained:
        # iterate over the three possible percentages
        for percentage in [25, 50, 75]:

            if (
                (task_and_file == "pc_04" and item == "pc_01")
                or (task_and_file == "pc_04" and item == "pc_02")
		or (task_and_file == "pc_04" and item == "pc_03" and percentage == 25)
		or (task_and_file == "pc_04" and item == "pc_03" and percentage == 50)
            ):
                continue
            X_train, y_train = get_training_concatenated_np_array(
                item, percentage, df_input_metadata, df_train_base
            )
            flaml_automl, flaml_automl_settings = get_set_classifier_flaml(
                task_and_file,  # 'base' dataset
                "gmm_random",  # gmm_random
                item,  # dataset that is going to be sampled
                percentage,  # percentage
            )
            timestamp_start_training = datetime.datetime.today()
            # fit AutoML system with the concatenated training set
            flaml_automl.fit(X_train=X_train, y_train=y_train, **flaml_automl_settings)

            timestamp_end_training = datetime.datetime.today()
            # it is going to predict the 'base' dataset and all the other sampled datasets
            for prediction_dataset in list_datasets_to_be_predicted:

                X_test, y_true = get_numpy_array_prediction(
                    prediction_dataset, percentage, df_input_metadata
                )

                timestamp_start_prediction = datetime.datetime.today()
                y_pred = flaml_automl.predict(X_test)
                y_pred = [str(x).lower() for x in y_pred]
                y_true = [str(x).lower() for x in y_true]
                timestamp_end_prediction = datetime.datetime.today()
                normalized_y_true = ",".join(y_true)
                normalized_y_pred = ",".join(y_pred)
                # it creates a str with the combination of the base dataset, concatenated dataset, percentage, and predicted dataset
                combination = (
                    "base_dataset_"
                    + task_and_file
                    + "_dataset_sampled_to_concat_"
                    + item
                    + "_percentage_"
                    + str(percentage)
                    + "_holdout_dataset_to_be_predicted_"
                    + prediction_dataset
                )
                print(f"combination: {combination}")

                (
                    time_history,
                    best_valid_loss_history,
                    valid_loss_history,
                    config_history,
                    metric_history,
                ) = get_output_from_log(
                    filename=flaml_automl_settings["log_file_name"],
                    time_budget=60 * 60 * SELECTED_WALL_TIME,
                )

                dict_partial_columns = {
                    "duration_training": [
                        timestamp_end_training - timestamp_start_training
                    ],
                    "duration_prediction": [
                        timestamp_end_prediction - timestamp_start_prediction
                    ],
                }
                dict_partial_columns["automl_system"] = ["FLAML"] * 1
                dict_partial_columns["dataset_compared"] = [prediction_dataset]
                dict_partial_columns["trained_dataset"] = [task_and_file] * 1
                dict_partial_columns["hold_out_f1_score"] = [
                    f1_score(y_true, y_pred, average="micro")
                ]
                dict_partial_columns["validation_best_score"] = [flaml_automl.best_loss]
                dict_partial_columns["validation_all_scores"] = [
                    best_valid_loss_history
                ]  # F1
                dict_partial_columns["5_fold_validation_scores"] = [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ] * 1
                dict_partial_columns["hold_out_y_true"] = [normalized_y_true]
                dict_partial_columns["hold_out_y_pred"] = [normalized_y_pred]
                dict_partial_columns["hold_out_y_pred_proba"] = [[0, 0, 0, 0, 0]] * 1
                dict_partial_columns["name_best_classifier"] = [
                    f"{flaml_automl.best_estimator} {str(flaml_automl.best_config)}"
                ]
                dict_partial_columns["method"] = ["gmm_random"] * 1
                dict_partial_columns["percentage"] = [percentage]
                dict_partial_columns["combination_trained"] = [combination]
                df_dataset_summary_part = pd.DataFrame(dict_partial_columns)
                results.write_summary(
                    df_dataset_summary_part,
                    f"part_flaml_100_combination_summary_gmm_random_{combination}.csv",
                )
except Exception as e:
    task_and_file = SELECTED_DATASET
    with open(
        output_path(f"flaml_error_100_combination_gmm_random_{task_and_file}.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"Error: {e}\n{traceback.format_exc()}")
