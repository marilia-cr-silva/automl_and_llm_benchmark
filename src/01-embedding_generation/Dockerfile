# Stage 01: generate DistilBERT [CLS] embeddings of a raw text CSV (TensorFlow, CPU/GPU as available).
# Build (from the project root):  docker build -f src/01-embedding_generation/Dockerfile -t amlb-embeddings .
# Run:  docker run --rm -v "$PWD/data:/data" -e RAW_FILENAME=hs_01_bin_0_1_train.csv amlb-embeddings
FROM python:3.9
WORKDIR /app
COPY src/01-embedding_generation/requirements.txt /app/requirements.txt
RUN pip install uv && \
    uv pip install -r requirements.txt --system
COPY pyproject.toml /app/pyproject.toml
COPY src/automl_and_llm_benchmark /app/src/automl_and_llm_benchmark
RUN uv pip install --no-deps --system /app
COPY src/01-embedding_generation/embedding_generation.py /app/stage/embedding_generation.py
ENV WORK_DIR=/data \
    COLLECTION="" \
    READ_DIR="" \
    WRITE_DIR="" \
    RAW_FILENAME=""
# WORK_DIR:     default /data, I/O root
# COLLECTION:   optional, empty = canonical datasets, pt = Portuguese datasets under WORK_DIR/pt
# READ_DIR:     default WORK_DIR (empty = unset), directory with the raw CSV
# WRITE_DIR:    default WORK_DIR (empty = unset), directory receiving the embeddings parquet
# RAW_FILENAME: required, raw CSV file name (';' separated, columns text,label)
ENTRYPOINT ["python3", "/app/stage/embedding_generation.py"]
