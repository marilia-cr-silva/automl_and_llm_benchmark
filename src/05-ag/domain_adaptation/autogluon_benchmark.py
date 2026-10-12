# %%
"""
This .py file runs the AutoGluon AutoML System on concatenated datasets (domain adaptation)

For the 'base' dataset DATASET, 100% of its training set is concatenated with the
25/50/75% GMM-sampled training instances of each related dataset (see
automl_and_llm_benchmark/static/dataset_relations.csv); one AutoGluon model is trained per
(related dataset, percentage), and it predicts the hold-out sets of every related
dataset and of the base dataset.

Environment variables:
    DATASET      - id of the 'base' dataset, e.g. "ed_03" (required)
    WALL_TIME    - AutoGluon training time limit per model, in hours (required)
    WORK_DIR     - directory with the inputs and receiving the outputs (default /data)
    NUM_THREADS  - BLAS/OpenMP threads (default 32)
    COLLECTION   - must be unset/empty: this stage only supports the canonical datasets

Inputs:  embeddings parquet files named in the packaged input_metadata CSVs, under
         WORK_DIR (stage 01 output); selected instances come from stage 02
Outputs: WORK_DIR/part_autogluon_summary_gmm_random_{combination}.csv (one per prediction)
         WORK_DIR/autogluon_summary_base_dataset_{DATASET}__dataset_sampled_to_concat_{item}_percentage_{p}/
         (AutoGluon model directories, replaced if they exist)
         WORK_DIR/autogluon_error_gmm_random_{DATASET}.txt (stack trace, only on error)

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

import datetime
import os
import warnings
import shutil
import traceback
from typing import Union

warnings.filterwarnings("ignore")

from automl_and_llm_benchmark.common import io, labels, paths, relations, results, selection
from automl_and_llm_benchmark.common.collection import require_canonical
from automl_and_llm_benchmark.common.env import require_env, require_int_env

require_canonical("05 AutoGluon domain adaptation")

SELECTED_DATASET = require_env("DATASET")
SELECTED_WALL_TIME = require_int_env("WALL_TIME")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from autogluon.tabular import TabularPredictor


# %%
def get_training_concatenated_autogluon(
    dataset_to_concat: str,
    selected_percentage: int,
    df_input: pd.DataFrame,
    df_base: pd.DataFrame,
) -> pd.DataFrame:
    """This function retrieves the sampled datasets and filters by the selected dataset with its percentage
        that are going to be concatenated with the 'base' dataset.
        After extracting the indices from the dataframe, only these rows are selected to concatenate the 'base' dataset.
        All the labels are replaced based on the dictionary with their correspondence.
        The concatenated training set is returned as a Pandas Dataframe with two columns (embeddings and labels)

    Args:
        dataset_to_concat (str): dataset that is sampled
        selected_percentage (int): the percentage of the data available in the sampled dataset
        df_input (pd.DataFrame): the dataframe that has the entire dataset that is going to be sampled
        df_base (pd.DataFrame): the 'base' dataset that is going to be used with 100% of its training set

    Returns:
       pd.DataFrame: the concatenated ('base' dataset + a different sampled dataset) training set where X includes embeddings and labels
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
    df_sampled_to_concat = selection.select_instances(
        df_sampled_to_concat, select_train_instance_idx
    )
    dict_replacements = labels.load_label_map(dataset_to_concat)
    # the labels need to be lower case before replacement with corresponding new labels
    df_sampled_to_concat["label"] = labels.normalize_labels(
        df_sampled_to_concat["label"]
    )
    df_sampled_to_concat["label"] = df_sampled_to_concat["label"].replace(
        dict_replacements
    )
    df_concat = pd.concat([df_base, df_sampled_to_concat], axis=0).reset_index(
        drop=True
    )

    X_train_embeddings = pd.DataFrame(np.array(df_concat["text"].tolist()))
    y_train_labels = pd.DataFrame(np.array(df_concat["label"].tolist()))
    y_train_labels.columns = ["label"]
    X_train_final = pd.concat([X_train_embeddings, y_train_labels], axis=1)

    print(
        f"\n before: {df_base['label'].unique()} after: {X_train_final['label'].unique()}"
    )

    return X_train_final


def get_df_prediction_autogluon(
    prediction_dataset_selected: str, selected_percentage: int, df_input: pd.DataFrame
) -> Union[pd.DataFrame, np.array]:
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
    dict_replacements = labels.load_label_map(prediction_dataset_selected)
    df_test_aux["label"] = labels.normalize_labels(df_test_aux["label"])
    df_test_aux["label"] = df_test_aux["label"].replace(dict_replacements)

    X_test_pd = pd.DataFrame(np.array(df_test_aux["text"].tolist()))
    y_true_array = np.array(df_test_aux["label"].tolist())
    return X_test_pd, y_true_array


# %%
df_input_metadata = relations.get_input_metadata()
df_parquet_files = df_input_metadata.query("dataset.str.contains(@SELECTED_DATASET)")
df_parquet_files = df_parquet_files.reset_index(drop=True)

try:
    task_and_file = SELECTED_DATASET
    # the 'base' is the dataset that uses 100% of its data and is combined with another sampled dataset
    df_train_base = io.read_embeddings(df_parquet_files["train"].iloc[0])
    dict_replacement = labels.load_label_map(task_and_file)
    # the labels need to be lower case before replacement with corresponding new labels
    df_train_base["label"] = labels.normalize_labels(df_train_base["label"])
    df_train_base["label"] = df_train_base["label"].replace(dict_replacement)
    df_train_base = df_train_base.reset_index(drop=True)

    list_datasets_to_be_trained, list_datasets_to_be_predicted = (
        relations.get_list_training_and_prediction(task_and_file)
    )

    print(f"list_datasets_to_be_trained{list_datasets_to_be_trained}")

    # iterate over datasets that are going to be sampled and concatenated with the 'base' dataset
    for item in list_datasets_to_be_trained:
        # iterate over the three possible percentages

        for percentage in [25, 50, 75]:

            if (
                (task_and_file == "ed_03")
                or (task_and_file == "fn_05")
                or (task_and_file == "hs_05")
                or (task_and_file == "pc_03")
                or (task_and_file == "pc_04")
                or (task_and_file == "pc_05" and item == "pc_01")
                or (task_and_file == "pc_05" and item == "pc_02")
                or (task_and_file == "pc_05" and item == "pc_03")
                or (task_and_file == "pc_05" and item == "pc_04")
                or (task_and_file == "pc_05" and item == "pc_06")
                or (task_and_file == "pc_05" and item == "pc_07")
                or (task_and_file == "pc_05" and item == "pc_08" and percentage == 25)
                or (task_and_file == "pc_05" and item == "pc_08" and percentage == 50)
                or (task_and_file == "pc_06")
                or (task_and_file == "pc_07" and item == "pc_01")
                or (task_and_file == "pc_07" and item == "pc_02")
                or (task_and_file == "pc_07" and item == "pc_03")
                or (task_and_file == "pc_07" and item == "pc_04")
                or (task_and_file == "pc_07" and item == "pc_05")
                or (task_and_file == "pc_07" and item == "pc_06")
                or (task_and_file == "pc_08" and item == "pc_01")
                or (task_and_file == "pc_08" and item == "pc_02")
                or (task_and_file == "pc_08" and item == "pc_03")
                or (task_and_file == "pc_08" and item == "pc_04")
                or (task_and_file == "pc_08" and item == "pc_05")
                or (task_and_file == "pc_08" and item == "pc_06")
                or (task_and_file == "pc_08" and item == "pc_07" and percentage == 25)
                or (task_and_file == "pc_09" and item == "pc_01")
                or (task_and_file == "pc_09" and item == "pc_02")
                or (task_and_file == "pc_09" and item == "pc_03")
                or (task_and_file == "pc_09" and item == "pc_04")
                or (task_and_file == "pc_09" and item == "pc_05")       
                or (task_and_file == "pc_09" and item == "pc_06")      
                or (task_and_file == "pc_09" and item == "pc_07")     
                or (task_and_file == "pc_09" and item == "pc_08" and percentage == 25) 
                or (task_and_file == "pc_09" and item == "pc_08" and percentage == 50)   
                or (task_and_file == "ua_01" and item == "ua_03")
                or (task_and_file == "ua_01" and item == "ua_04")
                or (task_and_file == "ua_01" and item == "ua_05")
                or (task_and_file == "ua_01" and item == "ua_06" and percentage == 25)
                or (task_and_file == "ua_01" and item == "ua_06" and percentage == 50)
                or (task_and_file == "ua_02" and item == "ua_01")
                or (task_and_file == "ua_02" and item == "ua_03")
                or (task_and_file == "ua_02" and item == "ua_04")
                or (task_and_file == "ua_02" and item == "ua_05")
                or (task_and_file == "ua_03")
                or (task_and_file == "ua_04")
                or (task_and_file == "ua_05")
                or (task_and_file == "ua_06")
            ):
                continue

            df_concatenated = pd.DataFrame()

            results_dir = (
                str(
                    paths.output_path(
                        f"autogluon_summary_base_dataset_{task_and_file}__dataset_sampled_to_concat_{item}_percentage_{str(percentage)}"
                    )
                )
                + os.sep
            )

            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)

            os.mkdir(results_dir)

            X_train = get_training_concatenated_autogluon(
                item, percentage, df_input_metadata, df_train_base
            )

            unique_labels = X_train["label"].unique()
            number_unique_label = len(unique_labels)

            if number_unique_label == 2:
                TYPE_OF_PROBLEM = "binary"
            else:
                TYPE_OF_PROBLEM = "multiclass"

            timestamp_start_training = datetime.datetime.today()
            # fit AutoML system with the concatenated training set
            predictor = TabularPredictor(
                label="label",
                problem_type=TYPE_OF_PROBLEM,
                eval_metric="f1_micro",
                path=results_dir,
            )
            print(f"isinstance test: {isinstance(X_train,pd.DataFrame)}")
            predictor.fit(
                X_train,
                time_limit=60 * 60 * SELECTED_WALL_TIME,
                presets="high_quality",
                num_gpus=1,
                ag_args_ensemble={"num_folds_parallel": 2},
            )
            timestamp_end_training = datetime.datetime.today()
            # it is going to predict the 'base' dataset and all the other sampled datasets
            for prediction_dataset in list_datasets_to_be_predicted:

                X_test, y_true = get_df_prediction_autogluon(
                    prediction_dataset, percentage, df_input_metadata
                )

                timestamp_start_prediction = datetime.datetime.today()
                y_pred = predictor.predict(X_test)
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
                predictor_leaderboard = predictor.leaderboard(extra_info=True)
                children_hyperparameters = str(
                    predictor_leaderboard.iloc[0]["child_hyperparameters_fit"]
                )
                hyperparameters_fitted = predictor_leaderboard.iloc[0][
                    "hyperparameters_fit"
                ]
                if hyperparameters_fitted:
                    hyperparameters_selected = str(hyperparameters_fitted)
                else:
                    hyperparameters_selected = str(
                        predictor_leaderboard.iloc[0]["hyperparameters"]
                    )
                list_hyperparameters_selected = []
                list_hyperparameters_selected.append(children_hyperparameters)
                list_hyperparameters_selected.append(hyperparameters_selected)
                str_hyperparameters = " ".join(list_hyperparameters_selected)
                best_validation_score = predictor_leaderboard.iloc[0]["score_val"]
                all_validation_metrics = (
                    predictor_leaderboard.dropna(subset=["score_val"])
                    .reset_index(drop=True)
                    .iloc[0]["score_val"]
                    .tolist()
                )
                best_model_name = predictor_leaderboard.iloc[0]["model"]
                name_best_classifier = best_model_name + " " + str_hyperparameters

                # the following lines create a dictionary that becomes a row in the dataset representing the each "combination"
                dict_partial_columns = {
                    "duration_training": [
                        timestamp_end_training - timestamp_start_training
                    ],
                    "duration_prediction": [
                        timestamp_end_prediction - timestamp_start_prediction
                    ],
                }
                dict_partial_columns["automl_system"] = ["AutoGluon"] * 1
                dict_partial_columns["dataset_compared"] = [prediction_dataset]
                dict_partial_columns["trained_dataset"] = [task_and_file] * 1
                dict_partial_columns["hold_out_f1_score"] = [
                    f1_score(y_true, y_pred, average="micro")
                ]
                dict_partial_columns["validation_best_score"] = [best_validation_score]
                dict_partial_columns["validation_all_scores"] = [
                    all_validation_metrics
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
                dict_partial_columns["name_best_classifier"] = [name_best_classifier]
                dict_partial_columns["method"] = ["gmm_random"] * 1
                dict_partial_columns["percentage"] = [percentage]
                dict_partial_columns["combination_trained"] = [combination]
                df_dataset_summary_part = pd.DataFrame(dict_partial_columns)
                results.write_summary(
                    df_dataset_summary_part,
                    f"part_autogluon_summary_gmm_random_{combination}.csv",
                )

except Exception as e:
    task_and_file = SELECTED_DATASET
    with open(
        paths.output_path(f"autogluon_error_gmm_random_{task_and_file}.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"Error: {e}\n{traceback.format_exc()}")
