# RORetrieval container (Apptainer)

GPU image for generating Romanian retrieval queries with **Gemma 3**, plus the
rest of the project stack (sentence-transformers, faiss, datasets).

| Component | Version | Why |
|---|---|---|
| CUDA | 12.4.1 (cudnn runtime) | broad driver compatibility (needs driver ≥ 550) |
| PyTorch | 2.6.0 (cu124) | required by vLLM 0.8.5 |
| vLLM | 0.8.5.post1 | continuous batching; first release line with Gemma 3 support |
| transformers | 4.51.3 | Gemma 3 needs ≥ 4.50; vLLM 0.8.5 needs the 4.x line |
| sentence-transformers | 4.1.0 (`<5`) | 5.x requires transformers ≥ 5, which breaks vLLM |

All versions sit in the first block of `%post` in
[roretrieval.def](roretrieval.def) — edit there if your cluster driver is older
(for CUDA 12.1 drivers switch `TORCH_INDEX` to `.../whl/cu121` and use the
matching base image).

The whole stack is installed in a **single pip call** on purpose: split across
several calls, a later package silently upgrades an earlier pin. A build-time
guard asserts `transformers` stayed on 4.x, so this fails during the build
rather than on a GPU node after queueing.

## 1. Build

Building needs root or fakeroot, so it usually happens **on your own machine**,
not on the FEP login node:

```bash
bash containers/build.sh                 # -> containers/roretrieval.sif (~9GB)
```

Other modes:

```bash
BUILD_MODE=sudo   bash containers/build.sh     # local root
BUILD_MODE=remote bash containers/build.sh     # Sylabs remote builder
IMAGE=/scratch/$USER/ro.sif bash containers/build.sh
```

Then copy it to the cluster:

```bash
rsync -avP containers/roretrieval.sif user@fep:/scratch/user/RORetrieval/containers/
```

## 2. Data on the cluster

The corpus is stored with git-lfs, so a fresh clone only contains pointer files:

```bash
git lfs install
git lfs pull
```

The generator prints a warning and skips any category file that is still an
unfetched pointer, so an empty run usually means this step was missed.

Alternatively, skip git-lfs entirely and fetch the corpus from the original
sources — compute nodes here have outbound internet:

```bash
mkdir -p logs
sbatch --partition=haswell scripts/slurm/download_data.sbatch
```

Run this as a batch job, not on the login node: Apptainer tears down the
squashfuse mount of the `.sif` when the session that started it ends, which
kills anything long-running detached with `nohup`.

## 3. Gemma access (one-time)

Gemma 3 is gated:

1. Accept the license at <https://huggingface.co/google/gemma-3-4b-it>
2. Create a read token at <https://huggingface.co/settings/tokens>
3. `cp containers/env.example containers/env.sh`, fill in `HF_TOKEN`, then
   `source containers/env.sh`

If compute nodes have no internet, cache the weights from the login node first:

```bash
bash scripts/download_model.sh
```

## 4. Run

On a GPU node (interactive):

```bash
apptainer exec --nv \
    --bind "$PWD":"$PWD" --pwd "$PWD" \
    containers/roretrieval.sif \
    python3 -m src.task1_queries.gemma_query_generation --max-docs 2000
```

Through SLURM (2-hour slot, the intended path):

```bash
mkdir -p logs
source containers/env.sh
sbatch --partition=<your-gpu-partition> scripts/slurm/generate_queries.sbatch
```

The job stops itself at `QGEN_TIME_BUDGET_MIN` (default 110 min) so it never
gets killed mid-write. Results stream to disk continuously; resubmitting the
same job continues where it left off (`--resume` skips finished `doc_id`s).

## 5. Sizing a 2-hour run

Throughput depends heavily on the GPU. Rough numbers for `gemma-3-4b-it` with
vLLM, 4 queries per document:

| GPU | ~docs / 2h |
|---|---|
| A100 80GB | 150k+ |
| A100 40GB / L40S | ~100k |
| V100 32GB (`DTYPE=float16`) | ~30k |
| RTX 2080Ti / T4 (`DTYPE=float16`, `MAX_MODEL_LEN=1536`) | ~10-15k |

`MAX_DOCS` defaults to 20000, split evenly across categories — a safe first run
on any of them. Raise it once you have measured the real rate (the job prints
`docs/h` at the end and writes it to `<output>.stats.json`).

## 6. Environment variables

Anything that must be visible **inside** the container needs the
`APPTAINERENV_` prefix — the SLURM script already forwards the important ones.

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | HuggingFace token for the gated Gemma repo |
| `HF_HOME` | weight cache (point at a large filesystem, not a small home quota) |
| `HF_HUB_OFFLINE=1` | use only cached weights (offline compute nodes) |
| `GEMMA_MODEL` | model id, default `google/gemma-3-4b-it` |
| `QGEN_BACKEND` | `auto` \| `vllm` \| `transformers` |
| `QGEN_OUTPUT` | output JSONL path |
| `QGEN_TIME_BUDGET_MIN` | wall-clock budget before a clean stop |
| `DTYPE` | `bfloat16` (Ampere+) or `float16` (V100/T4) |

## 7. Troubleshooting

**`torch.cuda.is_available()` is False** — the `--nv` flag is missing, or you
are on a login node without a GPU.

**vLLM fails at startup / CUDA error: no kernel image** — the driver is older
than CUDA 12.4. Either load a newer driver module or rebuild with the cu121
index. As a stopgap, `QGEN_BACKEND=transformers BATCH_SIZE=8` still works.

**OOM while loading the model** — lower `--gpu-memory-utilization` (0.80),
`--max-model-len` (1536) and `--batch-size`, or move to a smaller model.

**Permission denied writing a cache** — `HF_HOME`, `TRITON_CACHE_DIR` or
`XDG_CACHE_HOME` points somewhere read-only; set them to a scratch path and
re-export with the `APPTAINERENV_` prefix.

**Packages from `~/.local` leaking in** — already prevented by
`PYTHONNOUSERSITE=1` in `%environment`; if you override it, expect version
clashes with the container's torch.
