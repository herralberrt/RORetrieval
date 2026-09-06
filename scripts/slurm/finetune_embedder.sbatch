#!/bin/bash
#SBATCH --job-name=roq-ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=11:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# FINAL PHASE, step 2: fine-tune one model on one of our triplet sets.
#
#     MODEL=BAAI/bge-m3 SET=bm25 \
#         sbatch --partition=dgxa100 scripts/slurm/finetune_embedder.sbatch
#
# Four runs make the grid: {bge-m3, Qwen3-Embedding-8B} x {bm25, colbert}.
# One model per job rather than a loop, so a failure costs one run and the four
# can queue in parallel across the node's eight A100s.
#
# The wall clock is 11:30, just under the account's 12h MaxWall. Qwen3-8B with
# LoRA over ~77k rows does not finish an epoch in that; MAX_STEPS caps it, and
# both models get the same step budget so the comparison stays fair.
#
# Use dgxa100 - h200 never runs jobs from this account.

set -euo pipefail

PROJ_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJ_ROOT/containers/roretrieval.sif}"

MODEL="${MODEL:-BAAI/bge-m3}"
SET="${SET:-bm25}"                 # bm25 | colbert
BATCH="${BATCH:-}"                 # empty = per-model default below
MAX_STEPS="${MAX_STEPS:-2000}"
# LORA=1 forces the adapter path on a model that would otherwise train
# fully. Used to separate "the data caused the out-of-domain drop" from
# "the training recipe did": bge-m3 lost 0.097 nDCG on msmarco under a full
# fine-tune, Qwen lost less under LoRA, and only holding the data fixed
# while changing the recipe tells the two apart.
LORA="${LORA:-}"
EPOCHS="${EPOCHS:-1}"
SEQ="${SEQ:-512}"
OUTPUT="${OUTPUT:-}"

case "$SET" in
    bm25)    PREFIX="triplets_27b_bm25_hf" ;;
    colbert) PREFIX="triplets_27b_colbert_hf" ;;
    *) echo "ERROR: SET must be bm25 or colbert (got '$SET')" >&2; exit 1 ;;
esac
TRAIN="data/triplets/hf/${PREFIX}_train.parquet data/triplets/hf/${PREFIX}_eval.parquet"

# Batch size is the in-batch negative count, so it is a quality setting, not
# only a speed one. These are what fits an 80 GB A100 at 512 tokens: bge-m3 is
# 568M and trains fully, Qwen3-8B is LoRA plus gradient checkpointing.
if [[ -z "$BATCH" ]]; then
    case "$MODEL" in
        *Qwen3-Embedding-8B*) BATCH=8 ;;
        *) BATCH=64 ;;
    esac
fi
[[ -z "$OUTPUT" ]] && OUTPUT="models/$(basename "$MODEL" | tr "[:upper:]" "[:lower:]")-${SET}"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
[[ -z "$APPTAINER_BIN" ]] && { echo "ERROR: apptainer not available." >&2; exit 1; }
[[ ! -f "$IMAGE" ]] && { echo "ERROR: image not found: $IMAGE" >&2; exit 1; }

cd "$PROJ_ROOT"
mkdir -p logs models

export APPTAINERENV_PYTHONPATH="$PROJ_ROOT/.pylibs"
export APPTAINERENV_HF_HOME="${HF_HOME:-$PROJ_ROOT/.hf_cache}"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME:-$PROJ_ROOT/.hf_cache}"

echo "=================================================================="
echo " Job    : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Model  : $MODEL"
echo " Triplets: $SET  ->  $TRAIN"
echo " Output : $OUTPUT"
echo " Batch  : $BATCH   steps: $MAX_STEPS   seq: $SEQ"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$APPTAINER_BIN" exec --nv \
    --bind "$PROJ_ROOT":"$PROJ_ROOT" --pwd "$PROJ_ROOT" "$IMAGE" \
    python3 -m src.training.finetune_embedder \
        --model "$MODEL" \
        --train $TRAIN \
        --output "$OUTPUT" \
        --batch-size "$BATCH" \
        --epochs "$EPOCHS" \
        --max-steps "$MAX_STEPS" \
        --max-seq-length "$SEQ" \
        ${LORA:+--lora}

echo
echo "=================================================================="
echo " Done: $OUTPUT"
echo " Evaluate it with:"
echo "   ADAPTER=$OUTPUT MODELS=$MODEL TAG=ft-$SET \\"
echo "     sbatch --partition=dgxa100 scripts/slurm/evaluate_retrieval.sbatch"
echo " (bge-m3 trains fully, so pass MODELS=$OUTPUT and no ADAPTER instead)"
echo "=================================================================="
