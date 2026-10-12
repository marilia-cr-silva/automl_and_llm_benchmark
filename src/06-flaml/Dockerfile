# Stage 06: FLAML AutoML baseline on one dataset (reads embeddings, writes the metrics CSV and the FLAML log).
# Build (from the project root):  docker build -f src/06-flaml/Dockerfile -t amlb-flaml .
# Run:  docker run --rm -v "$PWD/data:/data" -e DATASET=hs_01 -e TEST_FILE=hs_01_bin_0_1_test_embeddings.parquet.brotli -e TRAIN_FILE=hs_01_bin_0_1_train_embeddings.parquet.brotli amlb-flaml
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
	apt-get install software-properties-common -y && \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt-get install python3.9 -y && \
	apt-get install python3.9-distutils -y && \
	apt-get install python3-pip -y && \
	apt-get install -y build-essential
RUN /bin/python3.9 -m pip install --upgrade pip setuptools
COPY src/06-flaml/requirements.txt /app/requirements.txt
RUN /bin/python3.9 -m pip install -r requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY src/automl_and_llm_benchmark /app/src/automl_and_llm_benchmark
RUN /bin/python3.9 -m pip install --no-deps /app
COPY src/06-flaml/flaml_benchmark_single.py /app/stage/flaml_benchmark_single.py
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
ENTRYPOINT ["/bin/python3.9", "/app/stage/flaml_benchmark_single.py"]
