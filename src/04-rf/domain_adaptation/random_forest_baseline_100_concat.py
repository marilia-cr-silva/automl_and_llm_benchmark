# %%
"""Random Forest (random search) baseline for domain adaptation by dataset concatenation.

For every "base" dataset in dataset_relations (static data), trains on 100% of the base
train split concatenated with the GMM-selected (25/50/75%) train instances of each related
dataset, and predicts the hold-out sets of the related datasets and of the base dataset.
Labels of every dataset are renamed to a common vocabulary (label maps of the canonical collection). Loops over all base datasets: it reads no
dataset variable.

Environment variables:
    WORK_DIR   - directory with the embeddings parquets and receiving the outputs (default /data)
    COLLECTION - must be unset/empty: this stage only supports the canonical datasets
Inputs (stage 01-embedding_generation, stage 02-gmm):
    embeddings parquets named in the static input_metadata/gmm_{25,50,75}.csv, under WORK_DIR
Outputs:
    WORK_DIR/random_forest_100_summary_<method>_<base dataset>.csv
    WORK_DIR/random_forest_100_error_summary_<method>_<base dataset>_<percentage>.txt (on failure)
"""
import warnings

warnings.filterwarnings("ignore")
import datetime
import traceback
from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from automl_and_llm_benchmark.common import io, results
from automl_and_llm_benchmark.common.collection import require_canonical
from automl_and_llm_benchmark.common.labels import load_label_map, normalize_labels, rename_labels
from automl_and_llm_benchmark.common.paths import output_path
from automl_and_llm_benchmark.common.random_forest import build_search
from automl_and_llm_benchmark.common.relations import get_input_metadata, get_list_training_and_prediction
from automl_and_llm_benchmark.common.resources import read_static_csv
from automl_and_llm_benchmark.common.selection import select_instances

require_canonical("04 random forest domain adaptation")

# %%
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
        df_input (pd.DataFrame): the df that has the entire dataset that is going to be sampled
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
    print("get_training_concatenated_np_array worked!")
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
        df_input (pd.DataFrame): df with dataset metadata (such as file name, instance selection method, percentage)

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

df_dataset_relations = read_static_csv("dataset_relations.csv")

datasets_to_train_set = set(df_dataset_relations["first_dataset"].tolist())
datasets_to_train_list = list(datasets_to_train_set)

# These result lists are created once and never reset: rows accumulate across the base
# datasets, so each summary CSV also contains the rows of the previously processed bases.
list_y_true = []
list_y_pred = []
list_percentage = []
list_f1_score_hold_out = []
list_evaluated_dataset = []
list_duration_prediction = []
list_duration_training = []
list_combination_datasets_percentage_trained = []
list_name_estimator = []
list_dict_best_config = []
list_name_best_classifier = []
list_best_validation_score = []
list_trained_datasets = []


for selected_dataset_item in datasets_to_train_list:
    df_parquet_files = df_input_metadata.query(
        "dataset.str.contains(@selected_dataset_item)"
    )
    df_parquet_files = df_parquet_files.drop_duplicates("dataset", ignore_index=True)

    array_parquet_files = np.array(df_parquet_files)
    number_pairs, _ = array_parquet_files.shape

    for counter in range(number_pairs):
        try:
            task_and_file = array_parquet_files[counter][-1]
            print(
                f"base: {task_and_file} parquet base: {array_parquet_files[counter][1]}"
            )
            # the 'base' is the dataset that uses 100% of its data and is combined with another sampled dataset
            df_train_base = io.read_embeddings(array_parquet_files[counter][1])
            dict_replacement = load_label_map(task_and_file)
            # the labels need to be lower case before replacement with corresponding new labels
            df_train_base["label"] = normalize_labels(df_train_base["label"])
            df_train_base["label"] = rename_labels(
                df_train_base["label"], dict_replacement
            )
            df_train_base = df_train_base.reset_index(drop=True)

            clf = build_search()

            list_datasets_to_be_trained, list_datasets_to_be_predicted = (
                get_list_training_and_prediction(task_and_file)
            )

            print(f"list_datasets_to_be_trained{list_datasets_to_be_trained}")

            # iterate over datasets that are going to be sampled and concatenated with the 'base' dataset
            for item in list_datasets_to_be_trained:
                # iterate over the three possible percentages
                for percentage in [25, 50, 75]:
                    X_train, y_train = get_training_concatenated_np_array(
                        item, percentage, df_input_metadata, df_train_base
                    )
                    timestamp_start_training = datetime.datetime.today()
                    # fit RF with the concatenated training set
                    random_search_clf = clf.fit(X_train, y_train)
                    timestamp_end_training = datetime.datetime.today()

                    # it is going to predict the 'base' dataset and all the other sampled datasets
                    for prediction_dataset in list_datasets_to_be_predicted:

                        X_test, y_true = get_numpy_array_prediction(
                            prediction_dataset, percentage, df_input_metadata
                        )

                        timestamp_start_prediction = datetime.datetime.today()
                        y_pred = random_search_clf.predict(X_test)
                        y_pred = [str(x).lower() for x in y_pred]
                        y_true = [str(x).lower() for x in y_true]
                        timestamp_end_prediction = datetime.datetime.today()
                        normalized_y_true = ",".join(y_true)
                        normalized_y_pred = ",".join(y_pred)
                        # true and predicted values are stored in a list of lists
                        list_y_true.append([normalized_y_true])
                        list_y_pred.append([normalized_y_pred])
                        # it calculates F1-Score micro for each prediction
                        list_f1_score_hold_out.append(
                            f1_score(y_true, y_pred, average="micro")
                        )
                        # it stores the predicted dataset
                        list_evaluated_dataset.append(prediction_dataset)
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
                        list_combination_datasets_percentage_trained.append(combination)
                        # duration training 'base' dataset + sampled dataset
                        list_duration_training.append(
                            (timestamp_end_training - timestamp_start_training)
                        )
                        # duration prediction dataset (with the same classes or with some classes of the 'base' dataset)
                        list_duration_prediction.append(
                            (timestamp_end_prediction - timestamp_start_prediction)
                        )
                        # percentage of the sampled dataset that was concatenated with the 'base' dataset
                        list_percentage.append(percentage)
                        # name best estimator
                        list_name_estimator.append("Random Forest with Random Search")

                        # name best hyperparameters RF
                        list_dict_best_config.append(clf.best_params_)
                        list_name_best_classifier.append(
                            f"'Random Forest with Random Search' {str(clf.best_params_)}"
                        )
                        list_best_validation_score.append(clf.best_score_)  # f1

                        list_trained_datasets.append(task_and_file)

            # the following lines create a dictionary that becomes a row in the dataset representing the each "combination"
            total_occurrences = len(list_f1_score_hold_out)
            dict_dataframe_columns = {
                "duration_training": list_duration_training,
                "duration_prediction": list_duration_prediction,
            }

            name_best_classifier = "Random Forest"

            dict_dataframe_columns["automl_system"] = [
                "Random Forest with Random Search"
            ] * total_occurrences
            dict_dataframe_columns["dataset_compared"] = list_evaluated_dataset
            dict_dataframe_columns["trained_dataset"] = list_trained_datasets
            dict_dataframe_columns["hold_out_f1_score"] = list_f1_score_hold_out
            dict_dataframe_columns["validation_best_score"] = list_best_validation_score
            dict_dataframe_columns["validation_all_scores"] = [
                [0, 0, 0, 0, 0]
            ] * total_occurrences

            dict_dataframe_columns["5_fold_validation_scores"] = [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ] * total_occurrences
            dict_dataframe_columns["hold_out_y_true"] = list_y_true
            dict_dataframe_columns["hold_out_y_pred"] = list_y_pred

            dict_dataframe_columns["hold_out_y_pred_proba"] = [
                [0, 0, 0, 0, 0]
            ] * total_occurrences
            dict_dataframe_columns["name_best_classifier"] = [
                name_best_classifier
            ] * total_occurrences
            dict_dataframe_columns["method"] = [
                array_parquet_files[counter][3]
            ] * total_occurrences
            dict_dataframe_columns["percentage"] = list_percentage
            dict_dataframe_columns["combination_trained"] = (
                list_combination_datasets_percentage_trained
            )
            df_dataset_summary = pd.DataFrame(dict_dataframe_columns)
            results.write_summary(
                df_dataset_summary,
                f"random_forest_100_summary_{array_parquet_files[counter][3]}_{task_and_file}.csv",
            )
        except Exception as e:
            task_and_file = array_parquet_files[counter][-1]
            with open(
                output_path(
                    f"random_forest_100_error_summary_{array_parquet_files[counter][3]}_{task_and_file}_{array_parquet_files[counter][4]}.txt"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(f"Error: {e}\n{traceback.format_exc()}")
