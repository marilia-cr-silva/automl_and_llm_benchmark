# Stage reference

One entry script and one Dockerfile per row (13 of each). Everything is configured with
environment variables (`docker run -e NAME=value`); there are no command-line arguments.
All paths below are relative to `WORK_DIR` (default `/data`, the mount point of your data
directory). See [pipeline.md](pipeline.md) for the data flow and a worked example.

Build every image from the **project root**, with the Dockerfile selected by `-f`:

```sh
docker build -f src/04-rf/Dockerfile.method -t amlb-rf-method .
```

The tags used below (`amlb-*`) are the ones suggested in each Dockerfile header; they are only
a convention. Variables marked `required` have an empty default in the Dockerfile and the
script stops with an error if they are unset or empty. `WORK_DIR` (default `/data`) and
`COLLECTION` are read by every stage and are not repeated in the "optional" column.

`COLLECTION` selects the dataset collection (see
[pipeline.md](pipeline.md#collections-canonical-and-pt)). Unset or empty is the canonical
collection (the default) and `pt` is the adjacent Portuguese one, whose files live under
`WORK_DIR/pt`; any other value is an error.

| Stages | `COLLECTION` |
|---|---|
| 01, 02, 03, 04/05/06 single and method | optional: empty = canonical, `pt` = all files under `WORK_DIR/pt` |
| 04/05/06 domain adaptation | must stay empty (canonical only; the script stops otherwise) |
| 07, 08 | required: `pt` (there is no canonical LLM file list yet; the script stops before loading the model otherwise) |

## Images and variables

| Stage | Script (`src/...`) | Dockerfile -> tag | Required | Optional (default) | GPU | Base image |
|---|---|---|---|---|---|---|
| 01 embeddings | `01-embedding_generation/embedding_generation.py` | `01-embedding_generation/Dockerfile` -> `amlb-embeddings` | `RAW_FILENAME` | `READ_DIR` (=`WORK_DIR`), `WRITE_DIR` (=`WORK_DIR`), `TRANSFORMERS_CACHE` (not declared; read by `transformers`) | not used by the image (no CUDA libraries installed; TensorFlow runs on CPU) | `python:3.9` |
| 02 GMM selection | `02-gmm/gmm_single.py` | `02-gmm/Dockerfile` -> `amlb-gmm` | `DATASET`, `FILENAME` | `NUM_THREADS` (16) | no | `python:3.9` |
| 03 LLM sampling | `03-sampling/sampling_datasets_to_llms.py` | `03-sampling/Dockerfile` -> `amlb-sampling` | `DATASET`, `FILENAME` | - | no | `python:3.9` |
| 04 RF, single | `04-rf/random_forest_baseline_single.py` | `04-rf/Dockerfile` -> `amlb-rf` | `DATASET`, `TRAIN_FILE`, `TEST_FILE` | - | no | `nvidia/cuda:12.3.1-runtime-ubuntu22.04` |
| 04 RF, method | `04-rf/random_forest_baseline.py` | `04-rf/Dockerfile.method` -> `amlb-rf-method` | `DATASET`, `FILENAME` | - | no | same as above |
| 04 RF, domain adaptation | `04-rf/domain_adaptation/random_forest_baseline_100_concat.py` | `04-rf/domain_adaptation/Dockerfile` -> `amlb-rf-da` | none (loops over every base dataset) | - | no | same as above |
| 05 AutoGluon, single | `05-ag/autogluon_benchmark_single.py` | `05-ag/Dockerfile` -> `amlb-ag` | `DATASET`, `TRAIN_FILE`, `TEST_FILE` | `NUM_THREADS` (32) | requested (`num_gpus=1`) | `autogluon/autogluon:0.8.2-cuda11.7-jupyter-ubuntu20.04-py3.9` |
| 05 AutoGluon, domain adaptation | `05-ag/domain_adaptation/autogluon_benchmark.py` | `05-ag/domain_adaptation/Dockerfile` -> `amlb-ag-da` | `DATASET`, `WALL_TIME` (hours, integer) | `NUM_THREADS` (32) | requested (`num_gpus=1`) | same as above |
| 06 FLAML, single | `06-flaml/flaml_benchmark_single.py` | `06-flaml/Dockerfile` -> `amlb-flaml` | `DATASET`, `TRAIN_FILE`, `TEST_FILE` | - | no | `nvidia/cuda:12.3.1-runtime-ubuntu22.04` + Python 3.9 (deadsnakes) |
| 06 FLAML, method | `06-flaml/flaml_benchmark.py` | `06-flaml/Dockerfile.method` -> `amlb-flaml-method` | `DATASET`, `FILENAME` | - | no | same as above |
| 06 FLAML, domain adaptation | `06-flaml/domain_adaptation/flaml_benchmark_baseline_100_concat.py` | `06-flaml/domain_adaptation/Dockerfile` -> `amlb-flaml-da` | `DATASET`, `WALL_TIME` (hours, integer) | - | no | same as above |
| 07 Llama 3.2 3B | `07-llama/final_checkpoint_llama_3_2.py` | `07-llama/Dockerfile` -> `amlb-llama` | `DATASET`, `COLLECTION` (`pt`) | `NUMBER_OF_EXAMPLES` (0), `LEAP` (0), `MODEL_DIR` (`/data/.llama/checkpoints/Llama3.2-3B-Instruct`), `NUM_THREADS` (16) | required | `nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |
| 08 Mistral Nemo | `08-mistral/mistral.py` | `08-mistral/Dockerfile` -> `amlb-mistral` | `DATASET`, `COLLECTION` (`pt`) | `NUMBER_OF_EXAMPLES` (0), `LEAP` (0), `MODEL_DIR` (`/data/mistral-nemo-instruct-2407`) | required | same as 07 |

Notes:

- 04, 05 and 06 use **pinned** per-stage requirements (`src/0N-*/requirements.txt`); the domain
  adaptation images reuse the stage-level file. Pins are not shared across stages (they conflict).
- 07 and 08 run through `torchrun --nproc_per_node=1 ... --master_port=273` (port 273,
  a privileged port number).
- `NUM_THREADS` sets `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `OMP_NUM_THREADS` unless
  those are already set. 07 fixes `OMP_NUM_THREADS` to 1. Stages without a `NUM_THREADS` entry
  do not configure threads.
- 04 and 06 images are built on a CUDA base but their scripts use no GPU.
- 01 downloads `distilbert-base-cased` from the HuggingFace hub at run time (nothing is baked
  into the image), so it needs network access on first use.

## Inputs and outputs (under `WORK_DIR`, or `WORK_DIR/pt` with `COLLECTION=pt`)

`<id>` is a dataset id such as `hs_01`; `<DATASET>` is the value of the env var;
`<method>` is the `method` column of the selected-instances CSV (always `gmm_random` here);
`<pct>` is the `percentage` column (25, 50, 75).

| Stage | Reads | Writes |
|---|---|---|
| 01 | `<READ_DIR>/<RAW_FILENAME>` (`;`-separated, columns `text`, `label`; rows with NaN are dropped) | `<WRITE_DIR>/<RAW_FILENAME up to its first '.'>_embeddings.parquet.brotli` (columns `text` = embedding vector, `label`) |
| 02 | `FILENAME`: train embeddings parquet | `gmm_random_<DATASET>_selected_instances.csv` (columns `test;train;selected_train_instance_idx;method;percentage`), `gmm_random_<DATASET>_value_counts.csv` |
| 03 | `FILENAME`: selected-instances CSV from 02; raw train CSV `<DATASET>_bin_train_0_1.csv` or `<DATASET>_multi_train.csv` (binary if the `train` column contains `_bin_`) | `<DATASET>_bin_llm.csv` or `<DATASET>_multi_llm.csv` (columns `text;label`; at most 10 examples per label, from the 25% selection) |
| 04 single | `TRAIN_FILE`, `TEST_FILE` (embeddings parquet) | `random_forest_summary_<DATASET>.csv`; `random_forest_error_<DATASET>.txt` on failure |
| 04 method | `FILENAME` (selected-instances CSV from 02) and the embeddings parquets named in its `train` / `test` columns | per CSV row: `random_forest_summary_<method>_<DATASET>_<pct>.csv`; `random_forest_error_<method>_<DATASET>_<pct>.txt` on failure |
| 04 DA | embeddings parquets named in the packaged `input_metadata/gmm_{25,50,75}.csv` | `random_forest_100_summary_<method>_<base>.csv` per base dataset; `random_forest_100_error_summary_<method>_<base>_<pct>.txt` on failure |
| 05 single | `TRAIN_FILE`, `TEST_FILE` | `autogluon_summary_<DATASET>.csv`; model directory `autogluon_<DATASET>/` (deleted and recreated); `autogluon_error_<DATASET>` (no `.txt`) on failure |
| 05 DA | same embeddings as 04 DA | `part_autogluon_summary_gmm_random_<combination>.csv` per prediction; model directories `autogluon_summary_base_dataset_<DATASET>__dataset_sampled_to_concat_<item>_percentage_<pct>/`; `autogluon_error_gmm_random_<DATASET>.txt` on failure |
| 06 single | `TRAIN_FILE`, `TEST_FILE` | `flaml_summary_<DATASET>.csv`, `flaml_automl_<DATASET>.log`; `flaml_error_<DATASET>.txt` on failure |
| 06 method | like 04 method | `flaml_summary_<method>_<DATASET>_<pct>.csv`, `flaml_automl_3_<DATASET>_4.log` (one shared name), `flaml_error_<method>_<DATASET>_<pct>.txt` on failure |
| 06 DA | same embeddings as 04 DA | `part_flaml_100_combination_summary_gmm_random_<combination>.csv`, `flaml_automl_base_<base>_concat_gmm_random_<item>_<pct>.log`, `flaml_error_100_combination_gmm_random_<DATASET>.txt` on failure |
| 07 | `<id>_bin_llm.csv` / `<id>_multi_llm.csv` (from 03) and the raw test CSV, both looked up by `DATASET` in the packaged `static/pt/zero_and_few_shot_files.csv`; model weights in `MODEL_DIR` (an absolute path, not moved under `WORK_DIR/pt`) | `LLAMA_3_2_3B_INSTRUCT_<DATASET>_predicted_result_<N>_shot_end_index_<i>.csv` (every 3000 instances), `..._<N>_shot_final_remaining.csv`; `error_LLAMA_3_2_3B_INSTRUCT_<DATASET>_predicted_result_<N>_shot.txt` on failure |
| 08 | same as 07; weights and `tekken.json` in `MODEL_DIR` | same names with `MISTRAL_NEMO_INSTRUCT_2407` instead of `LLAMA_3_2_3B_INSTRUCT` |

`<N>` is `NUMBER_OF_EXAMPLES`. The domain adaptation stages need no stage-02 output at run time:
the selected instance indices are inside the packaged metadata CSVs.

## Recommended `docker run` resource flags

The flags below are **site-specific recommendations** for running these images on a large
machine. Scale them to yours.

| Stage | CPUs | Memory / swap / shm | GPU |
|---|---|---|---|
| 01, 02, 04 | not set (02: `NUM_THREADS` default 16) | `--memory=1000g --memory-swap=1000g --shm-size=1000gb` | none needed |
| 03, 06 | `--cpus=16` | same as above | none needed |
| 05 | `--cpus=16` (script default `NUM_THREADS=32`: set `-e NUM_THREADS=16` if you limit CPUs); domain adaptation uses 32 threads | same as above | one GPU: `--gpus '"device=0"'` |
| 07 | not set | `--memory=500g --memory-swap=500g --shm-size=500gb` | one GPU |
| 08 | not set | `--memory=1000g --memory-swap=1000g --shm-size=1000gb` | one GPU |

Run the container as the invoking user so that files written into the bind mount are not
owned by root:

```sh
docker run --rm \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --memory=1000g --memory-swap=1000g --shm-size=1000gb \
  --gpus '"device=0"' \
  -v "$PWD/data:/data" \
  -e DATASET=hs_01 -e TRAIN_FILE=hs_01_bin_train_0_1_embeddings.parquet.brotli \
  -e TEST_FILE=hs_01_bin_test_0_1_embeddings.parquet.brotli amlb-ag
```

With `--user`, the process has no writable home directory, so any HOME-based cache fails.
`-e HOME=/tmp` covers that. Stage 01 also downloads the DistilBERT model into the HuggingFace
cache: point it at a persistent, writable location to avoid re-downloading, for example
`-e TRANSFORMERS_CACHE=/data/.hf_cache`. Stages 07 and 08 need the model weights **inside the
mounted data directory** at the default `MODEL_DIR`, or elsewhere if you mount that path and set
`MODEL_DIR`. Run them with `-e COLLECTION=pt`; their data is read from and written to
`<WORK_DIR>/pt`.
