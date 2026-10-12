# automl_and_llm_benchmark

Benchmark of text-classification methods on **Portuguese** datasets (doctoral research). Texts
are embedded with DistilBERT, training subsets are selected with a Gaussian mixture model, and
AutoML baselines (Random Forest, AutoGluon, FLAML) are compared with zero/few-shot LLMs
(Llama 3.2 3B Instruct, Mistral Nemo Instruct 2407).

Each stage is a standalone, Docker-driven step: a notebook-style script plus a Dockerfile,
configured only through environment variables, reading and writing one mounted data directory.

## Project layout

```
src/
  01-embedding_generation/   raw CSV -> DistilBERT [CLS] embeddings (parquet, brotli)
  02-gmm/                    GMM selection of 25/50/75% of the training instances
  03-sampling/               few-shot example CSV for the LLMs (from the 25% selection)
  04-rf/                     Random Forest baseline: single, method, domain_adaptation
  05-ag/                     AutoGluon baseline: single, domain_adaptation
  06-flaml/                  FLAML baseline: single, method, domain_adaptation
  07-llama/                  zero/few-shot classification with Llama 3.2 3B (GPU)
  08-mistral/                zero/few-shot classification with Mistral Nemo (GPU)
  automl_and_llm_benchmark/
    common/                  shared helpers (env, paths, io, labels, results, ...)
    static/                  packaged data: label maps, dataset relations, GMM input metadata
tests/                       pytest suite for automl_and_llm_benchmark.common
docs/                        pipeline.md (data flow, example, conventions), stages.md (reference)
data/                        runtime data mounted at /data (not shipped, see data/README.md)
```

## Quickstart

### A. Development setup (shared package and tests)

```sh
uv sync                       # installs automl_and_llm_benchmark + dev tools (pytest, ruff, scikit-learn)
uv run pytest
uv run ruff check src tests
```

This environment only covers `src/automl_and_llm_benchmark`. The stage environments are the pinned
`src/0N-*/requirements.txt` files (installed inside the Docker images). They conflict with each
other, so to run a stage script outside Docker use a throwaway virtualenv with the Python version
of that stage's image (3.9 for 01, 02, 03, 05, 06; 3.10 for 04, 07, 08):

```sh
uv venv --python 3.9 /tmp/venv-02
VIRTUAL_ENV=/tmp/venv-02 uv pip install -r src/02-gmm/requirements.txt
VIRTUAL_ENV=/tmp/venv-02 uv pip install --no-deps .
WORK_DIR=$PWD/data DATASET=hs_01 FILENAME=hs_01_bin_train_0_1_embeddings.parquet.brotli \
  /tmp/venv-02/bin/python src/02-gmm/gmm_single.py
```

### B. Run one stage with Docker

Build from the **project root** (the build context), selecting the Dockerfile with `-f`.
Mount your data directory at `/data` and pass everything else as environment variables:

```sh
docker build -f src/02-gmm/Dockerfile -t amlb-gmm .
docker run --rm -v "$PWD/data:/data" \
  -e DATASET=hs_01 \
  -e FILENAME=hs_01_bin_train_0_1_embeddings.parquet.brotli amlb-gmm
```

Inputs are read from, and outputs written to, `WORK_DIR` (default `/data`). The canonical datasets are the default; `-e COLLECTION=pt` selects the adjacent Portuguese datasets, under `WORK_DIR/pt` (required by stages 07/08). Required and optional
variables, inputs, outputs and recommended resource flags for each of the 13 images are in
[docs/stages.md](docs/stages.md); the data flow, a worked end-to-end example, file-name
conventions are in [docs/pipeline.md](docs/pipeline.md).
