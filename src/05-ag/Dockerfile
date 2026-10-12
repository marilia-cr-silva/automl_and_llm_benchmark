# Stage 05: AutoGluon on one dataset (reads train/test embeddings, writes summary CSV + model dir).
# Build (from the project root):  docker build -f src/05-ag/Dockerfile -t amlb-ag .
# Run:  docker run --rm -v "$PWD/data:/data" -e DATASET=hs_01 -e TRAIN_FILE=hs_01_bin_0_1_train_embeddings.parquet.brotli -e TEST_FILE=hs_01_bin_0_1_test_embeddings.parquet.brotli amlb-ag
FROM autogluon/autogluon:0.8.2-cuda11.7-jupyter-ubuntu20.04-py3.9
WORKDIR /app
RUN apt-get clean && \
	apt-get autoclean && \
	apt-get update --allow-unauthenticated && \
	apt-get install -y python3-pip
COPY src/05-ag/requirements.txt /app/requirements.txt
RUN pip install uv && \
	uv pip install -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match --system
COPY pyproject.toml /app/pyproject.toml
COPY src/automl_and_llm_benchmark /app/src/automl_and_llm_benchmark
RUN uv pip install --no-deps --system /app
COPY src/05-ag/autogluon_benchmark_single.py /app/stage/autogluon_benchmark_single.py
ENV WORK_DIR=/data \
    COLLECTION="" \
    NUM_THREADS=32 \
    DATASET="" \
    TRAIN_FILE="" \
    TEST_FILE=""
# WORK_DIR:    default /data, directory with the inputs and receiving the outputs
# COLLECTION:  optional, empty = canonical datasets, pt = Portuguese datasets under WORK_DIR/pt
# NUM_THREADS: default 32, BLAS/OpenMP threads
# DATASET:     required, dataset id (e.g. hs_01), used in the output file names
# TRAIN_FILE:  required, train embeddings parquet file name under WORK_DIR
# TEST_FILE:   required, test embeddings parquet file name under WORK_DIR
ENTRYPOINT ["python3", "/app/stage/autogluon_benchmark_single.py"]
