#!/bin/bash
#SBATCH --job-name=roq-triplets
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# TASK 2: mine hard negatives for the generated queries, inside the Apptainer image.
#
# The heavy part is embedding the corpus (117k documents) and the queries
# (~70k) with a MiniLM bi-encoder. On CPU that is ~30 min for the corpus alone
# - job 256311 hit its wall clock in the middle of it - so this asks for one
# GPU, where the same pass takes a couple of minutes. Any card works; the
# model is 120 MB, so `ml` or `xl` are as good as a DGX and queue faster.
#
# faiss-cpu does the search, hence the 16 CPUs.
#
# Submit:
#     mkdir -p logs
#     sbatch --partition=dgxa100 scripts/slurm/build_triplets.sbatch
#
# Override anything through the environment:
#     NEGATIVES=5 MIN_NEG_SIM=0.45 sbatch --partition=ml scripts/slurm/build_triplets.sbatch
#
# A rerun reuses the corpus embeddings cached next to the output
# (<output>.docvecs.npz), so tuning the similarity band is cheap: only the
# queries are re-embedded.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJ_ROOT/containers/roretrieval.sif}"

QUERIES="${QUERIES:-data/queries/queries_gemma3_27b.jsonl}"
TRIPLETS_OUTPUT="${TRIPLETS_OUTPUT:-data/triplets/triplets_27b.jsonl}"
ST_MODEL="${ST_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
NEGATIVES="${NEGATIVES:-3}"
CANDIDATES="${CANDIDATES:-50}"
MIN_NEG_SIM="${MIN_NEG_SIM:-0.35}"
MAX_NEG_SIM="${MAX_NEG_SIM:-0.92}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_QUERIES="${MAX_QUERIES:-0}"      # 0 = all; set it for a quick smoke run

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
if [[ ! -f "$PROJ_ROOT/$QUERIES" ]]; then
    echo "ERROR: no query file at $QUERIES - run generate_queries.sbatch first." >&2
    exit 1
fi

cd "$PROJ_ROOT"
mkdir -p logs "$(dirname "$TRIPLETS_OUTPUT")"

# The bi-encoder is ~120 MB, nothing like the 27B weights, so the home cache is
# always big enough - MODEL/HF_MODEL_GB keep hf_cache_dir.sh from sizing the
# check for Gemma and banishing it to node-local scratch.
MODEL="$ST_MODEL" HF_MODEL_GB=1 source scripts/hf_cache_dir.sh

export APPTAINERENV_HF_HOME="$HF_HOME"
if [[ -n "${HF_TOKEN:-}" ]]; then
    export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
fi
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=================================================================="
echo " Job          : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Image        : $IMAGE"
echo " Queries      : $QUERIES"
echo " Output       : $TRIPLETS_OUTPUT"
echo " Encoder      : $ST_MODEL"
echo " Negatives    : $NEGATIVES from $CANDIDATES candidates in [$MIN_NEG_SIM, $MAX_NEG_SIM]"
echo " HF cache     : $HF_HOME"
echo "=================================================================="
nvidia-smi || echo "(no GPU visible - this will fall back to CPU and be slow)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
"$APPTAINER_BIN" exec --nv \
    --bind "$PROJ_ROOT":"$PROJ_ROOT" \
    --bind "$HF_HOME":"$HF_HOME" \
    --pwd "$PROJ_ROOT" \
    "$IMAGE" \
    python3 -m src.task2_triplets.build_triplets \
        --queries "$QUERIES" \
        --output "$TRIPLETS_OUTPUT" \
        --model "$ST_MODEL" \
        --negatives "$NEGATIVES" \
        --candidates "$CANDIDATES" \
        --min-neg-similarity "$MIN_NEG_SIM" \
        --max-neg-similarity "$MAX_NEG_SIM" \
        --batch-size "$BATCH_SIZE" \
        --max-queries "$MAX_QUERIES"

echo "=================================================================="
echo " Done. Inspect the triplets with:"
echo "   python3 -m src.task2_triplets.inspect_triplets --input $TRIPLETS_OUTPUT"
echo "=================================================================="
