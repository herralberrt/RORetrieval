#!/bin/bash
#SBATCH --job-name=roq-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# FINAL PHASE, step 1: what bge-m3 and Qwen3-Embedding-8B score before any
# fine-tuning, on our test split and on ro-msmarco-divided.
#
#     sbatch --partition=dgxa100 scripts/slurm/evaluate_retrieval.sbatch
#
# Use dgxa100 - h200 never runs jobs from this account (see
# build_triplets_colbert.sbatch for the diagnosis).
#
# Numbers are comparable *within* a dataset, not across the two. Our test split
# carries ~3.7 negatives per query, ro-msmarco-divided ~29, so for a similar
# query count msmarco's corpus is roughly seven times larger and its retrieval
# task correspondingly harder. Compare model against model and before against
# after, never our column against msmarco's.
#
# MODELS and DATASETS can be overridden; ADAPTER evaluates a fine-tuned LoRA:
#     ADAPTER=models/qwen3-bm25 MODELS=Qwen/Qwen3-Embedding-8B \
#         sbatch --partition=dgxa100 scripts/slurm/evaluate_retrieval.sbatch

set -euo pipefail

PROJ_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJ_ROOT/containers/roretrieval.sif}"

MODELS="${MODELS:-BAAI/bge-m3 Qwen/Qwen3-Embedding-8B}"
DATASETS="${DATASETS:-data/triplets/hf/triplets_27b_bm25_hf_test.parquet data/triplets/hf/triplets_27b_colbert_hf_test.parquet alina0195/ro-msmarco-divided}"
OUTPUT="${OUTPUT:-results/retrieval_eval.jsonl}"
BATCH="${BATCH:-32}"
MAX_QUERIES="${MAX_QUERIES:-0}"
ADAPTER="${ADAPTER:-}"
TAG="${TAG:-zeroshot}"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
[[ -z "$APPTAINER_BIN" ]] && { echo "ERROR: apptainer not available." >&2; exit 1; }
[[ ! -f "$IMAGE" ]] && { echo "ERROR: image not found: $IMAGE" >&2; exit 1; }

cd "$PROJ_ROOT"
mkdir -p logs results

# peft is not in the image; it is installed --no-deps into .pylibs so the
# container's own torch/transformers keep winning. Installing it with deps
# pulls a second torch and breaks everything.
export APPTAINERENV_PYTHONPATH="$PROJ_ROOT/.pylibs"
export APPTAINERENV_HF_HOME="${HF_HOME:-$PROJ_ROOT/.hf_cache}"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME:-$PROJ_ROOT/.hf_cache}"

echo "=================================================================="
echo " Job     : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Models  : $MODELS"
echo " Datasets: $DATASETS"
echo " Output  : $OUTPUT   (tag: $TAG)"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

for model in $MODELS; do
    for dataset in $DATASETS; do
        short_m="$(basename "$model")"
        short_d="$(basename "${dataset%.parquet}")"
        echo
        echo "──────── $short_m  ×  $short_d"
        "$APPTAINER_BIN" exec --nv \
            --bind "$PROJ_ROOT":"$PROJ_ROOT" --pwd "$PROJ_ROOT" "$IMAGE" \
            python3 -m src.evaluation.evaluate_retrieval \
                --model "$model" \
                --dataset "$dataset" \
                --split test \
                --name "${TAG}::${short_m}::${short_d}" \
                --output "$OUTPUT" \
                --batch-size "$BATCH" \
                --max-queries "$MAX_QUERIES" \
                ${ADAPTER:+--adapter "$ADAPTER"} \
            || echo "  ✗ failed: $short_m × $short_d (continuing)"
    done
done

echo
echo "=================================================================="
echo " Done. Results appended to $OUTPUT"
echo "=================================================================="
