# Stage 03: sample the few-shot/LLM example CSV from the 25% GMM selection of one dataset.
# Build (from the project root):  docker build -f src/03-sampling/Dockerfile -t amlb-sampling .
# Run:  docker run --rm -v "$PWD/data:/data" -e DATASET=hs_01 -e FILENAME=gmm_random_hs_01_selected_instances.csv amlb-sampling
FROM python:3.9
WORKDIR /app
COPY src/03-sampling/requirements.txt /app/requirements.txt
RUN pip install uv && \
	uv pip install -r requirements.txt --system
COPY pyproject.toml /app/pyproject.toml
COPY src/automl_and_llm_benchmark /app/src/automl_and_llm_benchmark
RUN uv pip install --no-deps --system /app
COPY src/03-sampling/sampling_datasets_to_llms.py /app/stage/sampling_datasets_to_llms.py
ENV WORK_DIR=/data \
    COLLECTION="" \
    DATASET="" \
    FILENAME=""
# WORK_DIR:   default /data, directory with the inputs and receiving the outputs
# COLLECTION: optional, empty = canonical datasets, pt = Portuguese datasets under WORK_DIR/pt
# DATASET:    required, dataset id (e.g. hs_01)
# FILENAME:   required, selected-instances CSV name under WORK_DIR (output of stage 02)
ENTRYPOINT ["python3", "/app/stage/sampling_datasets_to_llms.py"]
