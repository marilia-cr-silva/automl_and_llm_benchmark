# Stage 05 domain adaptation: AutoGluon on a 'base' dataset concatenated with sampled related datasets.
# Build (from the project root):  docker build -f src/05-ag/domain_adaptation/Dockerfile -t amlb-ag-da .
# Run:  docker run --rm -v "$PWD/data:/data" -e DATASET=ed_01 -e WALL_TIME=2 amlb-ag-da
FROM autogluon/autogluon:0.8.2-cuda11.7-jupyter-ubuntu20.04-py3.9
WORKDIR /app
RUN apt-get clean
RUN apt-get autoclean
RUN apt-get update --allow-unauthenticated && \
    apt-get install -y python3-pip
COPY src/05-ag/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
COPY pyproject.toml /app/pyproject.toml
COPY src/automl_and_llm_benchmark /app/src/automl_and_llm_benchmark
RUN pip install --no-deps /app
COPY src/05-ag/domain_adaptation/autogluon_benchmark.py /app/stage/autogluon_benchmark.py
ENV WORK_DIR=/data \
    COLLECTION="" \
    NUM_THREADS=32 \
    DATASET="" \
    WALL_TIME=""
# WORK_DIR:    default /data, directory with the inputs and receiving the outputs
# COLLECTION:  must stay empty: this stage only supports the canonical datasets
# NUM_THREADS: default 32, BLAS/OpenMP threads
# DATASET:     required, id of the 'base' dataset (e.g. ed_01)
# WALL_TIME:   required, AutoGluon training time limit per model, in hours (integer)
ENTRYPOINT ["python3", "/app/stage/autogluon_benchmark.py"]
