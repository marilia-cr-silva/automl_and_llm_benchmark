# Stage 04: Random Forest (random search) baseline on one dataset (reads embeddings, writes the metrics CSV).
# Build (from the project root):  docker build -f src/04-rf/Dockerfile -t amlb-rf .
# Run:  docker run --rm -v "$PWD/data:/data" -e DATASET=hs_01 -e TEST_FILE=hs_01_bin_0_1_test_embeddings.parquet.brotli -e TRAIN_FILE=hs_01_bin_0_1_train_embeddings.parquet.brotli amlb-rf
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y build-essential && \
    apt-get install -y software-properties-common && \
    apt-get clean
COPY src/04-rf/requirements.txt /app/requirements.txt
RUN pip install uv && \
	uv pip install -r requirements.txt --system
COPY pyproject.toml /app/pyproject.toml
COPY src/automl_and_llm_benchmark /app/src/automl_and_llm_benchmark
RUN uv pip install --no-deps --system /app
COPY src/04-rf/random_forest_baseline_single.py /app/stage/random_forest_baseline_single.py
ENV WORK_DIR=/data \
    COLLECTION="" \
    DATASET="" \
    TEST_FILE="" \
    TRAIN_FILE=""
# WORK_DIR:   default /data, directory with the inputs and receiving the outputs
# COLLECTION: optional, empty = canonical datasets, pt = Portuguese datasets under WORK_DIR/pt
# DATASET:    required, dataset id (e.g. hs_01), used in the output file names
# TEST_FILE:  required, test embeddings parquet file name under WORK_DIR
# TRAIN_FILE: required, train embeddings parquet file name under WORK_DIR
ENTRYPOINT ["python3", "/app/stage/random_forest_baseline_single.py"]
