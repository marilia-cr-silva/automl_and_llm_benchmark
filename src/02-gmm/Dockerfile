# Stage 02: select 25/50/75% of the training instances with a GMM (reads embeddings, writes CSVs).
# Build (from the project root):  docker build -f src/02-gmm/Dockerfile -t amlb-gmm .
# Run:  docker run --rm -v "$PWD/data:/data" -e DATASET=hs_01 -e FILENAME=hs_01_bin_0_1_train_embeddings.parquet.brotli amlb-gmm
FROM python:3.9
WORKDIR /app
COPY src/02-gmm/requirements.txt /app/requirements.txt
RUN pip install uv && \
	uv pip install -r requirements.txt --system
COPY pyproject.toml /app/pyproject.toml
COPY src/automl_and_llm_benchmark /app/src/automl_and_llm_benchmark
RUN uv pip install --no-deps --system /app
COPY src/02-gmm/gmm_single.py /app/stage/gmm_single.py
ENV WORK_DIR=/data \
    COLLECTION="" \
    NUM_THREADS=16 \
    DATASET="" \
    FILENAME=""
# WORK_DIR:    default /data, directory with the inputs and receiving the outputs
# COLLECTION:  optional, empty = canonical datasets, pt = Portuguese datasets under WORK_DIR/pt
# NUM_THREADS: default 16, BLAS/OpenMP threads
# DATASET:     required, dataset id (e.g. hs_01), used in the output file names
# FILENAME:    required, train embeddings parquet file name under WORK_DIR
ENTRYPOINT ["python3", "/app/stage/gmm_single.py"]
