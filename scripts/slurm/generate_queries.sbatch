#!/bin/bash
#SBATCH --job-name=roq-gemma
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Gemma 3 query generation inside the Apptainer image, sized for a 2-hour slot.
#
# Submit (partition name differs per cluster - check `sinfo`):
#     mkdir -p logs
#     sbatch --partition=<gpu-partition> scripts/slurm/generate_queries.sbatch
#
# Override anything through the environment:
#     MAX_DOCS=50000 GEMMA_MODEL=google/gemma-3-12b-it \
#         sbatch --partition=gpu scripts/slurm/generate_queries.sbatch
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

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-3-4b-it}"
QGEN_OUTPUT="${QGEN_OUTPUT:-data/queries/queries_gemma3.jsonl}"
QGEN_BACKEND="${QGEN_BACKEND:-vllm}"
QGEN_TIME_BUDGET_MIN="${QGEN_TIME_BUDGET_MIN:-110}"
MAX_DOCS="${MAX_DOCS:-20000}"
QUERIES_PER_DOC="${QUERIES_PER_DOC:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DTYPE="${DTYPE:-bfloat16}"          # float16 on GPUs without bf16 (V100, T4)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

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
mkdir -p logs "$(dirname "$QGEN_OUTPUT")" "$HF_HOME"

# Pass variables through to the container.
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}"
export APPTAINERENV_QGEN_TIME_BUDGET_MIN="$QGEN_TIME_BUDGET_MIN"
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=================================================================="
echo " Job          : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Image        : $IMAGE"
echo " Model        : $GEMMA_MODEL"
echo " Output       : $QGEN_OUTPUT"
echo " Doc budget   : $MAX_DOCS   |  time budget: ${QGEN_TIME_BUDGET_MIN} min"
echo " HF cache     : $HF_HOME"
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
        --time-budget-min "$QGEN_TIME_BUDGET_MIN" \
        --resume

echo "=================================================================="
echo " Done. Inspect the run with:"
echo "   python3 -m src.task1_queries.analyze_query_metrics --input $QGEN_OUTPUT"
echo "=================================================================="
