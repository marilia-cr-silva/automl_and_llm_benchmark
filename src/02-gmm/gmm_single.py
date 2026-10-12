"""Select 25/50/75% of the training instances of one dataset with a Gaussian mixture model.

Environment variables:
    WORK_DIR  - directory with the inputs and receiving the outputs (default /data)
    COLLECTION - "pt" reads and writes under WORK_DIR/pt (default: unset = WORK_DIR)
    NUM_THREADS - BLAS/OpenMP threads (default 16)
    DATASET   - dataset id, used in the output file names (required)
    FILENAME  - train embeddings parquet file under WORK_DIR, from stage 01 (required)

Input:  <FILENAME> (e.g. hs_01_bin_0_1_train_embeddings.parquet.brotli)
Output: gmm_random_<DATASET>_selected_instances.csv
        (columns test;train;selected_train_instance_idx;method;percentage)
        gmm_random_<DATASET>_value_counts.csv
"""

# %%
from automl_and_llm_benchmark.common.threads import configure_threads

configure_threads(16)

import random
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

from automl_and_llm_benchmark.common import io, paths
from automl_and_llm_benchmark.common.env import require_env

dataset = require_env("DATASET")
filename = require_env("FILENAME")
work_dir = paths.work_dir()

print(f"{work_dir=} {filename=} {dataset=} ")

PROBABILITY_THRESHOLD = 0.1

# %%
df = io.read_embeddings(filename)

percent_25 = round(len(df) * 0.25)
percent_50 = round(len(df) * 0.50)
percent_75 = round(len(df) * 0.75)

list_length_percentage = [percent_25, percent_50, percent_75]
number_clusters = len(df["label"].unique())
np_embeddings = np.array(df["text"].tolist())

gmm = GaussianMixture(
    n_components=number_clusters, init_params="k-means++", random_state=42
)
gmm.fit(np_embeddings)
cluster_labels = gmm.predict_proba(np_embeddings)

cluster_labels_df = pd.DataFrame(cluster_labels)

df = df.reset_index(drop=True)
cluster_labels_df = cluster_labels_df.reset_index(drop=True)
df = pd.concat([df, cluster_labels_df], axis=1)

# There were inconsistencies when concatenating both DataFrames
if df.isna().any().any():
    raise ValueError("Inconsistencies between raw and GMM DataFrame")

list_indices = cluster_labels_df.columns.tolist()
df["max_prob"] = df[list_indices].max(axis=1)
df["min_prob"] = df[list_indices].min(axis=1)
df["max_diff"] = df["max_prob"] - df["min_prob"]
dict_elements_per_quartile = {}
dict_value_counts = {}
dict_value_counts["100"] = dict(df.label.value_counts())

list_select_indices_label = []
list_grouped_selected_indices = []

for sample_size in list_length_percentage:
    cluster_size = round(sample_size / len(list_indices))
    list_select_indices_label = []
    list_underrepresented_labels = []

    for name_column in list_indices:
        list_index = df[
            (df[name_column] == df["max_prob"])
            & (df["max_prob"] >= PROBABILITY_THRESHOLD)
        ].index.tolist()  # defining probability threshold
        if cluster_size <= len(list_index):
            random.seed(42)
            sampled_instances = random.sample(list_index, cluster_size)
            print(
                f"Valid: cluster size: {cluster_size} and list (length) {len(list_index)}"
            )
        else:
            print(
                f"Invalid: cluster size: {cluster_size} and list (length) {len(list_index)}"
            )
            sampled_instances = list_index
            list_underrepresented_labels.append(name_column)

        list_select_indices_label.extend(sampled_instances)
    list_mod = list_indices
    list_all_selected_percent = []
    while (
        list_underrepresented_labels
    ):  # sample again so that we reach the desired percentage
        set_list_indices = set(list_mod)
        set_under = set(list_underrepresented_labels)
        list_mod = list(set_list_indices - set_under)
        percentage = list_length_percentage.index(sample_size)
        df_mod = df[~df.index.isin(list_select_indices_label)]
        new_sample_size = sample_size - len(list_select_indices_label)
        new_cluster_size = round(new_sample_size / len(list_mod))
        list_underrepresented_labels = []

        for item_column in list_mod:
            list_index = df_mod[
                (df[item_column] == df["max_prob"])
                & (df["max_prob"] >= PROBABILITY_THRESHOLD)
            ].index.tolist()

            if new_cluster_size <= len(list_index):
                random.seed(42)
                sampled_instances = random.sample(list_index, new_cluster_size)
                print(
                    f"Valid: cluster size: {new_cluster_size} and list (length) {len(list_index)}"
                )
            else:
                print(
                    f"Invalid: cluster size: {new_cluster_size} and list (length) {len(list_index)}"
                )
                list_underrepresented_labels.append(item_column)
                sampled_instances = list_index

            list_select_indices_label.extend(sampled_instances)

    list_grouped_selected_indices.append(list_select_indices_label)

df_selected_instances = pd.DataFrame()
for idx, pct in enumerate(["25", "50", "75"]):
    list_all_selected_percent = list_grouped_selected_indices[idx]
    list_all_selected_percent.sort()
    df_filtered = df.iloc[list_all_selected_percent]
    dict_value_counts[pct] = dict(df_filtered.label.value_counts())
    dict_elements_per_quartile[pct] = list_all_selected_percent

    df_pct = pd.DataFrame()

    df_pct["selected_train_instance_idx"] = [list_all_selected_percent]
    df_pct["percentage"] = pct
    df_selected_instances = pd.concat([df_selected_instances, df_pct], axis=0)


df_selected_instances["test"] = filename.replace("train", "test")
df_selected_instances["train"] = filename
df_selected_instances["dataset"] = dataset
df_selected_instances["method"] = "gmm_random"

df_value_counts = pd.DataFrame(dict_value_counts)
io.write_csv(
    df_value_counts,
    f"gmm_random_{dataset}_value_counts.csv",
    index=True,
)

df_selected_instances = df_selected_instances[
    ["test", "train", "selected_train_instance_idx", "method", "percentage"]
]

io.write_csv(
    df_selected_instances,
    f"gmm_random_{dataset}_selected_instances.csv",
)
