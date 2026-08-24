# RORetrieval container (Apptainer)

GPU image for generating Romanian retrieval queries with **Gemma 3 27B**, plus
the rest of the project stack (sentence-transformers, faiss, datasets).

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

Gemma 3 is gated, and the license is accepted **per repository** — accepting it
for `gemma-3-4b-it` does not cover `gemma-3-27b-it`:

1. Accept the license at <https://huggingface.co/google/gemma-3-27b-it>
2. Create a read token at <https://huggingface.co/settings/tokens>
3. `cp containers/env.example containers/env.sh`, fill in `HF_TOKEN`, then
   `source containers/env.sh`

### Where the weights go

The bf16 27B checkpoint is **~55 GiB**, and a student home on
fep.grid.pub.ro has a hard 50 GiB CephFS quota:

```bash
getfattr -n ceph.quota.max_bytes ~     # 53687091200
```

So the 27B weights cannot live in the home cache. `scripts/hf_cache_dir.sh`
(sourced by the SLURM job and by `download_model.sh`) checks the quota and
falls back to node-local scratch — `/tmp/$USER/hf`, on a 1.6 TB local disk —
when the model does not fit. That cache is per node and does not survive
between jobs, so the weights are re-fetched each run; the dgxa100 and dgxh100
nodes have outbound internet and `HF_HUB_ENABLE_HF_TRANSFER=1` parallelises the
download: a measured cold 27B fetch on dgxa100 took **155 s** (~355 MB/s,
against ~70 MB/s single-stream), so it costs ~2% of a 110-minute budget.

The relocated cache has no `token` file in it, so the script also carries
`~/.cache/huggingface/token` over as `HF_TOKEN` — without that a gated download
fails with a 401.

Only when `HF_HOME` points somewhere big enough (or for smaller models) is
pre-downloading from the login node useful:

```bash
bash scripts/download_model.sh                       # default: 27B
bash scripts/download_model.sh google/gemma-3-4b-it  # fits in home
```

## 4. Run

On a GPU node (interactive):

```bash
apptainer exec --nv \
    --bind "$PWD":"$PWD" --pwd "$PWD" \
    containers/roretrieval.sif \
    python3 -m src.task1_queries.gemma_query_generation --max-docs 2000
```

On fep.grid.pub.ro the `student` account can submit to **dgxa100**
(8× A100-SXM4-80GB) and **dgxh100** (8× H100); `h200`, `hd`, `ml` and
`sprmcrogpu` are reserved for other accounts, and `xl` (P100) / `haswell`
(CPU-only) are too small for 27B. Check what is free with:

```bash
sinfo -O partition:14,nodelist:22,gres:26,gresused:30,statelong:12
```

Through SLURM (2-hour slot, the intended path):

```bash
mkdir -p logs
source containers/env.sh
sbatch --partition=dgxa100 scripts/slurm/generate_queries.sbatch
```

The job stops itself at `QGEN_TIME_BUDGET_MIN` (default 110 min) so it never
gets killed mid-write. Results stream to disk continuously; resubmitting the
same job continues where it left off (`--resume` skips finished `doc_id`s).

## 5. Sizing a 2-hour run

`gemma-3-27b-it` in bf16 needs ~55 GB for weights alone, so it takes one 80 GB
card (or two 40 GB cards with `TENSOR_PARALLEL=2 --gres=gpu:2`). The SLURM
script sums the VRAM of the allocated GPUs and refuses to start if there is not
enough, rather than dying in an OOM after the download.

| Model | VRAM (bf16) | Fits |
|---|---|---|
| `gemma-3-27b-it` | ~63 GB incl. KV cache | A100 80GB, H100, H200; 2× 40GB with TP=2 |
| `gemma-3-12b-it` | ~33 GB | A100 40GB and up |
| `gemma-3-4b-it` | ~17 GB | most GPUs |

Throughput with vLLM, 4 queries per document:

| GPU | 4B: ~docs / 2h | 27B: ~docs / 2h |
|---|---|---|
| A100 80GB | 150k+ | ~25k (measured) |
| A100 40GB / L40S | ~100k | needs TP=2 |
| V100 32GB (`DTYPE=float16`) | ~30k | does not fit |
| RTX 2080Ti / T4 (`DTYPE=float16`, `MAX_MODEL_LEN=1536`) | ~10-15k | does not fit |

The 27B number is measured: 17,453 documents at 4 queries each in 1:24:37 of
generation on one A100-SXM4-80GB — **12,374 docs/h**, 3.44 docs/s. Add ~5 min of
fixed startup (~2.5 min download into node-local scratch, ~1 min load, ~1.5 min
`torch.compile` plus CUDA graph capture, or ~15 s once `~/.cache/vllm` is warm).

vLLM reports `Maximum concurrency: 9.76x` for 2048-token requests, which looks
alarming next to `--batch-size 256`. It is not the binding constraint: these
prompts run ~600-700 tokens in and ~150 out, so far more than ten fit in the
KV cache at once. Lowering `MAX_MODEL_LEN` to buy concurrency is not worth the
restart.

`MAX_DOCS` defaults to 20000, split evenly across categories: each category is
capped at `ceil(MAX_DOCS / categories)` documents so one large outlet cannot
dominate the set. Categories smaller than that cap contribute everything they
have, so the run can finish **below** `MAX_DOCS` without hitting the time
budget — a 20000-document run yielded 17,453, because `aleph` (761), `recipes`
(818), `evz` (905), `realitatea` (1170) and `cotidianul` (1487) are all smaller
than the 1539 cap. Check `budget_reached` in `<output>.stats.json` to tell the
two cases apart: `false` means the corpus ran out, not the clock.

To go deeper into the large categories (`summarization` has 65k documents,
`adevarul` 10.6k, `stories` 12.5k), raise `MAX_DOCS` or set `--per-category`
directly.

## 6. Environment variables

Anything that must be visible **inside** the container needs the
`APPTAINERENV_` prefix — the SLURM script already forwards the important ones.

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | HuggingFace token for the gated Gemma repo |
| `HF_HOME` | weight cache (point at a large filesystem, not a small home quota) |
| `HF_HUB_OFFLINE=1` | use only cached weights (offline compute nodes) |
| `GEMMA_MODEL` | model id, default `google/gemma-3-27b-it` |
| `QGEN_BACKEND` | `auto` \| `vllm` \| `transformers` |
| `QGEN_OUTPUT` | output JSONL path |
| `QGEN_TIME_BUDGET_MIN` | wall-clock budget before a clean stop |
| `DTYPE` | `bfloat16` (Ampere+) or `float16` (V100/T4) |
| `TENSOR_PARALLEL` | split the model across N GPUs (needs `--gres=gpu:N`) |
| `GPU_MEM_UTIL` | vLLM memory fraction, default 0.90 |

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
