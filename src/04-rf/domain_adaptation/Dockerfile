# Stage 04 domain adaptation: Random Forest on concatenated datasets (loops over every base dataset; reads embeddings, writes the metrics CSVs).
# Build (from the project root):  docker build -f src/04-rf/domain_adaptation/Dockerfile -t amlb-rf-da .
# Run:  docker run --rm -v "$PWD/data:/data" amlb-rf-da
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get clean && \
    apt-get autoclean && \
    apt-get update --allow-unauthenticated && \
    apt-get install -y python3-pip && \
    apt-get install -y build-essential && \
    apt-get install -y software-properties-common && \
    apt-get clean
COPY src/04-rf/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY src/automl_and_llm_benchmark /app/src/automl_and_llm_benchmark
RUN pip install --no-deps /app
COPY src/04-rf/domain_adaptation/random_forest_baseline_100_concat.py /app/stage/random_forest_baseline_100_concat.py
ENV WORK_DIR=/data \
    COLLECTION=""
# WORK_DIR:   default /data, directory with the embeddings parquets and receiving the outputs
# COLLECTION: must stay empty: this stage only supports the canonical datasets
ENTRYPOINT ["python3", "/app/stage/random_forest_baseline_100_concat.py"]
