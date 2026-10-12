# Stage 07: zero/few-shot classification with Llama 3.2 3B Instruct (GPU, torchrun).
# Build (from the project root):  docker build -f src/07-llama/Dockerfile -t amlb-llama .
# Run:  docker run --rm -v "$PWD/data:/data" -e COLLECTION=pt -e DATASET=<id> -e NUMBER_OF_EXAMPLES=<n> amlb-llama
# /data must also hold the Llama checkpoint (see MODEL_DIR); the stage 03 / test CSVs are read from /data/pt.
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && \
    apt-get install -y gnupg ca-certificates && \
    apt-get update --allow-unauthenticated && \
    apt-get install -y python3-pip && \
    apt-get install -y python3-setuptools && \
    apt-get install -y python3-apt && \
    apt-get install -y build-essential && \
    apt-get clean

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

COPY src/07-llama/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt --resume-retries 30

COPY pyproject.toml /app/pyproject.toml
COPY src/automl_and_llm_benchmark /app/src/automl_and_llm_benchmark
RUN pip install --no-deps /app

COPY src/07-llama/final_checkpoint_llama_3_2.py /app/stage/final_checkpoint_llama_3_2.py

# Runtime configuration (all variables read by the script)
ENV WORK_DIR=/data \
    COLLECTION="" \
    NUM_THREADS=16 \
    MODEL_DIR=/data/.llama/checkpoints/Llama3.2-3B-Instruct \
    NUMBER_OF_EXAMPLES=0 \
    LEAP=0 \
    DATASET=""

# WORK_DIR           default /data: base data directory (inputs and outputs live in WORK_DIR/pt)
# COLLECTION         required: set to pt (there is no canonical LLM file list yet)
# NUM_THREADS        default 16: BLAS threads (OpenMP is fixed to 1 by the script)
# MODEL_DIR          default /data/.llama/checkpoints/Llama3.2-3B-Instruct: Llama checkpoint folder
# NUMBER_OF_EXAMPLES default 0: examples per label in the prompt (0 = zero-shot)
# LEAP               default 0: index of the first test instance to predict (resume)
# DATASET            required: dataset id (ed_01, fn_01, hs_01, pc_01, sd_01 ... sd_06)

ENTRYPOINT ["torchrun", "--nproc_per_node=1", "--nnodes=1", "--node_rank=0", "--master_addr=127.0.0.1", "--master_port=273", "/app/stage/final_checkpoint_llama_3_2.py"]
