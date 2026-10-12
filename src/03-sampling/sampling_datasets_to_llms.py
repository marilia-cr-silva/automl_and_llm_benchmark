"""Sample the few-shot/LLM example set of one dataset from its 25% GMM selection.

Environment variables:
    WORK_DIR - directory with the inputs and receiving the outputs (default /data)
    COLLECTION - "pt" reads and writes under WORK_DIR/pt (default: unset = WORK_DIR)
    DATASET  - dataset id (required)
    FILENAME - selected-instances CSV under WORK_DIR, from stage 02 (required)

Inputs:  <FILENAME> (gmm_random_<DATASET>_selected_instances.csv)
         <DATASET>_bin_train_0_1.csv or <DATASET>_multi_train.csv (raw train CSV, by task type)
Output:  <DATASET>_bin_llm.csv or <DATASET>_multi_llm.csv (columns text;label)
"""

# %%
import re
import pandas as pd

from automl_and_llm_benchmark.common import io, paths
from automl_and_llm_benchmark.common.env import require_env
from automl_and_llm_benchmark.common.labels import normalize_labels
from automl_and_llm_benchmark.common.selection import parse_instance_indices

# The LLM examples are built from the 25% selection of stage 02 only.
LLM_SAMPLING_PERCENTAGE = 25
# At most this many examples are kept per label (longest texts first).
MAX_EXAMPLES_PER_LABEL = 10

# %%
def number_alphabetic_characters(string: str) -> int:
    """This function counts the number of characters a-z
    It aims to 'penalize' instances that have more special characters
    More alphabetic characters may imply more context,
    which is essential for prompt engineering/ in-context learning

    Args:
        string (str): text of an instance

    Returns:
        int: number of alphabetic characters
    """

    list_words = string.split(" ")
    lower_words = [item.lower() for item in list_words]
    uncased_string = " ".join(lower_words)
    all_characters = re.findall("[a-z]", uncased_string)
    return len(all_characters)


work_dir = paths.work_dir()
dataset = require_env("DATASET")
filename = require_env("FILENAME")

print(f"{work_dir=} {filename=} {dataset=}")

# %%
selected_instances_gmm_df = io.read_csv(filename)
selected_instances_gmm_df = selected_instances_gmm_df[
    selected_instances_gmm_df["percentage"] == LLM_SAMPLING_PERCENTAGE
]
selected_instances_gmm_df["is_binary"] = selected_instances_gmm_df["train"].apply(
    lambda x: True if "_bin_" in x else False
)
selected_instances_gmm_df["part_final_name"] = selected_instances_gmm_df[
    "is_binary"
].apply(lambda x: "_bin_train_0_1.csv" if x == True else "_multi_train.csv")
selected_instances_gmm_df["dataset"] = dataset
selected_instances_gmm_df["final_name"] = (
    selected_instances_gmm_df["dataset"] + selected_instances_gmm_df["part_final_name"]
)
selected_instances_gmm_df = selected_instances_gmm_df.drop(
    ["test", "train", "method", "percentage", "part_final_name", "is_binary"], axis=1
)

# %%
all_datasets_list = list(selected_instances_gmm_df["dataset"])

# %%
for dataset_name in all_datasets_list:
    selected_row = selected_instances_gmm_df[
        selected_instances_gmm_df["dataset"] == dataset_name
    ]
    df_train = io.read_csv(selected_row["final_name"].iloc[0])

    # the labels need to be lower case before replacement with corresponding new labels
    df_train["label"] = normalize_labels(df_train["label"])

    df_train = df_train.dropna().reset_index(drop=True)

    select_train_instance_idx = list(selected_row["selected_train_instance_idx"])[0]
    select_train_instance_idx = parse_instance_indices(select_train_instance_idx)
    df_train_selected = df_train.loc[select_train_instance_idx]
    
    df_train_selected["number_alphabetic_characters"] = df_train_selected["text"].apply(
        number_alphabetic_characters
    )

    list_all_labels = list(df_train_selected["label"].unique())
    few_shot_learning_examples_df = pd.DataFrame()
    for label in list_all_labels:
        df_train_selected_filtered = df_train_selected.query("label==@label")
        df_train_selected_filtered = df_train_selected_filtered.sort_values(
            by=["number_alphabetic_characters"], ascending=False
        ).reset_index(drop=True)
        if len(df_train_selected_filtered) >= MAX_EXAMPLES_PER_LABEL:
            df_train_selected_filtered = df_train_selected_filtered.iloc[
                :MAX_EXAMPLES_PER_LABEL, :
            ]
            df_train_selected_filtered = df_train_selected_filtered.loc[
                :, ["text", "label"]
            ]
            few_shot_learning_examples_df = pd.concat(
                [few_shot_learning_examples_df, df_train_selected_filtered], axis=0
            )
        else:
            df_train_selected_filtered = df_train_selected_filtered.loc[
                :, ["text", "label"]
            ]
            few_shot_learning_examples_df = pd.concat(
                [few_shot_learning_examples_df, df_train_selected_filtered], axis=0
            )
    few_shot_learning_examples_df = few_shot_learning_examples_df.reset_index(drop=True)
    io.write_csv(
        few_shot_learning_examples_df,
        f"{'_'.join(selected_row['final_name'].iloc[0].split('_')[:3])}_llm.csv",
    )
    print(f"{dataset_name} was saved!")
