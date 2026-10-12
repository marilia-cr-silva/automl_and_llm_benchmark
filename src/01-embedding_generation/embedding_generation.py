"""Generate DistilBERT [CLS] embeddings (distilbert-base-cased, TensorFlow) of a raw text CSV.

Environment variables:
    WORK_DIR     - default for READ_DIR and WRITE_DIR (default /data)
    COLLECTION   - "pt" makes the READ_DIR/WRITE_DIR defaults WORK_DIR/pt (default: unset = WORK_DIR)
    READ_DIR     - directory with the raw CSV (default WORK_DIR)
    WRITE_DIR    - directory receiving the embeddings (default WORK_DIR)
    RAW_FILENAME - raw CSV file name, ';' separated with columns text,label (required)
    TRANSFORMERS_CACHE - optional, HuggingFace model cache directory (honored natively by
                         transformers; pass it with `docker run -e`)

Input:  <READ_DIR>/<RAW_FILENAME>
Output: <WRITE_DIR>/<RAW_FILENAME without extension>_embeddings.parquet.brotli
        (columns text = embedding vector, label)
"""

# %%
from pathlib import Path

import pandas as pd
from transformers import (
    DistilBertTokenizer,
    TFDistilBertModel,
)
from tqdm import tqdm

from automl_and_llm_benchmark.common import io, paths
from automl_and_llm_benchmark.common.env import get_env, require_env

tqdm.pandas()

# %%
NAME_TEXT_COLUMN = "text"
NAME_TARGET_COLUMN = "label"

MAX_SEQUENCE_LENGTH = 512
MODEL_NAME = "distilbert-base-cased"
TOKENIZER = DistilBertTokenizer.from_pretrained(MODEL_NAME)
BERT_BASE_MODEL = TFDistilBertModel.from_pretrained(MODEL_NAME)


def tokenization(sentences, max_length=MAX_SEQUENCE_LENGTH, padding="max_length"):
    return TOKENIZER(
        sentences,
        truncation=True,
        padding=padding,
        max_length=max_length,
        return_tensors="tf",
    )


def get_embedding_token_cls(row: pd.Series) -> list:
    text_instance = row[NAME_TEXT_COLUMN]
    text_instance = str(text_instance)
    list_tokens = tokenization(text_instance, padding="max_length")
    base_model_output = BERT_BASE_MODEL(list_tokens)
    cls_token = base_model_output.last_hidden_state[0][0].numpy().tolist()

    return cls_token


def generate_bert_base_cased(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    "For text classification tasks, BERT takes the final hidden state h of the first token [CLS] as the representation of the whole sequence."
    Source: https://arxiv.org/pdf/1905.05583.pdf
    """
    tqdm.pandas(desc="Processing")
    df_dataset["cls"] = df_dataset.progress_apply(get_embedding_token_cls, axis=1)
    df_dataset = df_dataset[["cls", NAME_TARGET_COLUMN]]
    df_dataset.columns = ["text", "label"]
    return df_dataset


# %%
read_dir = get_env("READ_DIR", str(paths.work_dir()))
write_dir = get_env("WRITE_DIR", str(paths.work_dir()))
filename = require_env("RAW_FILENAME")

print(f"{read_dir=} {write_dir=} {filename=}")

df_loaded = io.read_csv(Path(read_dir) / filename)
df_loaded = df_loaded.dropna()

result = generate_bert_base_cased(df_loaded)

output_filename = f'{filename.split(".")[0]}_embeddings.parquet.brotli'
io.write_embeddings(result, Path(write_dir) / output_filename)
