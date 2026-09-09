#!/bin/bash
#SBATCH --job-name=roq-answerable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Does each generated query have its answer in its own positive passage?
#
#     sbatch --partition=dgxa100 scripts/slurm/validate_answerability.sbatch
#
# The judge defaults to gemma-3-4b-it because it is the only capable instruct
# model already cached and the account has ~8 GB of quota left - a 9B or 12B
# would have to be downloaded first. It shares a family with the 27B that wrote
# the questions, so read the result as a floor on the problem rather than an
# independent measurement.
#
# Use dgxa100; h200 never runs jobs from this account.

set -euo pipefail

PROJ_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJ_ROOT/containers/roretrieval.sif}"

QUERIES="${QUERIES:-data/queries/queries_gemma3_27b.jsonl}"
OUTPUT="${OUTPUT:-results/answerability.jsonl}"
MODEL="${MODEL:-google/gemma-3-4b-it}"
SAMPLE="${SAMPLE:-2000}"
CONTROL="${CONTROL:-none}"
BATCH="${BATCH:-64}"
MAX_LEN="${MAX_LEN:-4096}"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
[[ -z "$APPTAINER_BIN" ]] && { echo "ERROR: apptainer not available." >&2; exit 1; }
[[ ! -f "$IMAGE" ]] && { echo "ERROR: image not found: $IMAGE" >&2; exit 1; }

cd "$PROJ_ROOT"
mkdir -p logs results

# gemma-3-4b-it sits in the default HF cache from the task-1 experiments, not
# in the project one, so point HF_HOME there and avoid re-downloading 8 GB into
# a quota that has about that much left.
export APPTAINERENV_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false

echo "=================================================================="
echo " Job    : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Judge  : $MODEL"
echo " Queries: $QUERIES   (sample: $SAMPLE)"
echo " Output : $OUTPUT"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$APPTAINER_BIN" exec --nv \
    --bind "$PROJ_ROOT":"$PROJ_ROOT" --bind "$HOME/.cache":"$HOME/.cache" \
    --pwd "$PROJ_ROOT" "$IMAGE" \
    python3 -m src.task1_queries.validate_answerability \
        --queries "$QUERIES" \
        --output "$OUTPUT" \
        --model "$MODEL" \
        --sample "$SAMPLE" \
        --batch-size "$BATCH" \
        --max-model-len "$MAX_LEN" \
        --control "$CONTROL"

echo "=================================================================="
