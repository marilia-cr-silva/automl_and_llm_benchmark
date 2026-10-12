# data/

Runtime data directory. Everything here other than this README is runtime data that is not shipped with the
project and is never copied into Docker images (see `.dockerignore`).

## Convention

Stage containers mount a host directory at `/data` and run with `WORK_DIR=/data`:

```
docker run -v "$PWD/data:/data" -e WORK_DIR=/data -e DATASET=ed_01 ... <image>
```

That single directory holds both inputs and outputs of every stage:

| Content | Naming / format |
|---|---|
| Raw datasets | CSV, `;` separator, columns `text,label`; the token order in the file names differs between stages, see "name patterns by stage" in `docs/pipeline.md` |
| Embeddings (stage 01) | `*_embeddings.parquet.brotli` |
| GMM selections (stage 02) | CSV with `selected_train_instance_idx`, `method`, `percentage`, ... |
| LLM example files (stage 03) | few-shot / `zero_and_few_shot_files.csv`-style CSVs |
| Outputs (stages 04-08) | metrics and LLM prediction files |

Dataset ids: `ed_01`, `fn_01`, `hs_01`, `pc_01`, `sd_01..06` (see `docs/pipeline.md`).

With `-e COLLECTION=pt` (the adjacent Portuguese datasets, required by stages 07/08) all of the above is
read from and written to the `pt/` subdirectory instead; canonical runs use the directory itself.
