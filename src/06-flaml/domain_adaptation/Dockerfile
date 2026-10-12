# Stage 06 domain adaptation: FLAML on a base dataset concatenated with each related dataset (reads embeddings, writes the metrics CSVs and FLAML logs).
# Build (from the project root):  docker build -f src/06-flaml/domain_adaptation/Dockerfile -t amlb-flaml-da .
# Run:  docker run --rm -v "$PWD/data:/data" -e DATASET=ed_03 -e WALL_TIME=1 amlb-flaml-da
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get clean && \
    apt-get autoclean && \
    apt-get update --allow-unauthenticated && \
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
COPY src/06-flaml/domain_adaptation/flaml_benchmark_baseline_100_concat.py /app/stage/flaml_benchmark_baseline_100_concat.py
ENV WORK_DIR=/data \
    COLLECTION="" \
    DATASET="" \
    WALL_TIME=""
# WORK_DIR:   default /data, directory with the embeddings parquets and receiving the outputs
# COLLECTION: must stay empty: this stage only supports the canonical datasets
# DATASET:    required, base dataset id (e.g. ed_03)
# WALL_TIME:  required, FLAML time budget per trained model, in hours (integer)
ENTRYPOINT ["/bin/python3.9", "/app/stage/flaml_benchmark_baseline_100_concat.py"]
