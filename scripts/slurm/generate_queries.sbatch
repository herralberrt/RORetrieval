#!/bin/bash
#SBATCH --job-name=roq-gemma
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Gemma 3 27B query generation inside the Apptainer image, sized for a 2-hour slot.
#
# The bf16 27B weights are ~55 GiB, so one GPU with >=80 GB works (A100 80GB,
# H100, H200). On fep.grid.pub.ro the `student` account can use dgxa100
# (8x A100-SXM4-80GB) and dgxh100 (8x H100); h200/hd/ml/sprmcrogpu are
# restricted to other accounts.
#
# --mem is 128G because the weights pass through host memory on the way to the
# GPU: a measured 27B run peaked at 59.5 GB RSS, so the old 32G would have been
# OOM-killed during load.
#
# Submit:
#     mkdir -p logs
#     sbatch --partition=dgxa100 scripts/slurm/generate_queries.sbatch
#
# Override anything through the environment:
#     MAX_DOCS=50000 GEMMA_MODEL=google/gemma-3-12b-it \
#         sbatch --partition=dgxa100 scripts/slurm/generate_queries.sbatch
#
# Two 40 GB cards instead of one 80 GB card:
#     TENSOR_PARALLEL=2 sbatch --partition=<part> --gres=gpu:2 \
#         scripts/slurm/generate_queries.sbatch
#
# The generator stops itself at QGEN_TIME_BUDGET_MIN (default 110 min) and
# flushes results continuously, so the job never gets killed mid-write.
# Rerun the same command to continue: --resume skips finished documents.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJ_ROOT/containers/roretrieval.sif}"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-3-27b-it}"
QGEN_OUTPUT="${QGEN_OUTPUT:-data/queries/queries_gemma3_27b.jsonl}"
QGEN_BACKEND="${QGEN_BACKEND:-vllm}"
QGEN_TIME_BUDGET_MIN="${QGEN_TIME_BUDGET_MIN:-110}"
MAX_DOCS="${MAX_DOCS:-20000}"
QUERIES_PER_DOC="${QUERIES_PER_DOC:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DTYPE="${DTYPE:-bfloat16}"          # float16 on GPUs without bf16 (V100, T4)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
APPTAINER_BIN="$(command -v apptainer || command -v singularity)"

if [[ -z "$APPTAINER_BIN" ]]; then
    echo "ERROR: apptainer/singularity not available on this node." >&2
    exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: image not found: $IMAGE  (build it with containers/build.sh)" >&2
    exit 1
fi

cd "$PROJ_ROOT"
mkdir -p logs "$(dirname "$QGEN_OUTPUT")"

# Picks a cache big enough for the weights (home quota, else node-local scratch).
source scripts/hf_cache_dir.sh

# Fail before the 55 GiB download rather than after it: vLLM needs room for the
# weights plus KV cache, and an OOM here costs the whole allocation.
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
           | awk '{s+=$1} END {print s+0}')"
NEED_GB=$(( HF_MODEL_GB + 8 ))
if (( VRAM_MB > 0 )) && (( VRAM_MB / 1024 < NEED_GB )); then
    echo "ERROR: $GEMMA_MODEL needs ~${NEED_GB} GB of VRAM, this job has $(( VRAM_MB / 1024 )) GB." >&2
    echo "       Use --partition=dgxa100 (A100 80GB), or request more cards with" >&2
    echo "       TENSOR_PARALLEL=N --gres=gpu:N, or pick a smaller GEMMA_MODEL." >&2
    exit 1
fi

# Pass variables through to the container.
export APPTAINERENV_HF_HOME="$HF_HOME"
# Only forward a token that actually exists: an empty HF_TOKEN inside the
# container would shadow the one stored by `huggingface-cli login`. Written as
# an `if` on purpose - a bare `[[ ... ]] && export` is a failing command under
# `set -e` when the token is unset, which killed the job right here.
if [[ -n "${HF_TOKEN:-}" ]]; then
    export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
fi
export APPTAINERENV_QGEN_TIME_BUDGET_MIN="$QGEN_TIME_BUDGET_MIN"
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Parallel shard downloads: single-stream HF gives ~70 MB/s, so a cold 27B
# cache would otherwise eat ~13 min of the time budget.
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "=================================================================="
echo " Job          : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Image        : $IMAGE"
echo " Model        : $GEMMA_MODEL  (tensor-parallel: $TENSOR_PARALLEL)"
echo " Output       : $QGEN_OUTPUT"
echo " Doc budget   : $MAX_DOCS   |  time budget: ${QGEN_TIME_BUDGET_MIN} min"
echo " HF cache     : $HF_HOME"
echo " VRAM         : $(( VRAM_MB / 1024 )) GB across $(nvidia-smi -L 2>/dev/null | wc -l) GPU(s)"
echo "=================================================================="
nvidia-smi || true

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
"$APPTAINER_BIN" exec --nv \
    --bind "$PROJ_ROOT":"$PROJ_ROOT" \
    --bind "$HF_HOME":"$HF_HOME" \
    --pwd "$PROJ_ROOT" \
    "$IMAGE" \
    python3 -m src.task1_queries.gemma_query_generation \
        --model "$GEMMA_MODEL" \
        --backend "$QGEN_BACKEND" \
        --output "$QGEN_OUTPUT" \
        --max-docs "$MAX_DOCS" \
        --queries-per-doc "$QUERIES_PER_DOC" \
        --batch-size "$BATCH_SIZE" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --time-budget-min "$QGEN_TIME_BUDGET_MIN" \
        --resume

echo "=================================================================="
echo " Done. Inspect the run with:"
echo "   python3 -m src.task1_queries.analyze_query_metrics --input $QGEN_OUTPUT"
echo "=================================================================="
