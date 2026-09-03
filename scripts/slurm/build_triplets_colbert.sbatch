#!/bin/bash
#SBATCH --job-name=roq-colbert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# TASK 2 (late-interaction variant): mine hard negatives with MaxSim.
#
# Needs a GPU, unlike the BM25 job: the corpus is encoded token-by-token and the
# whole tensor is held in device memory (87k docs x 128 tokens x 384 dims in
# fp16 is ~8.6 GB), so the rerank is one einsum with no host round-trip.
#
#     sbatch --partition=h200 scripts/slurm/build_triplets_colbert.sbatch
#
# Calibration first, always. The score band below is NOT the BM25 band - MaxSim
# sums cosines, so its ratios sit high and close together (p5 0.61, p50 0.82,
# p95 0.95 on a small real subset). Run with MAX_QUERIES set, read the reported
# distribution, then set MIN_RATIO/MAX_RATIO and run the full pass:
#
#     MAX_QUERIES=3000 TRIPLETS_OUTPUT=data/triplets/calib_colbert.jsonl \
#         sbatch --partition=h200 scripts/slurm/build_triplets_colbert.sbatch

set -euo pipefail

PROJ_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJ_ROOT/containers/roretrieval.sif}"

QUERIES="${QUERIES:-data/queries/queries_gemma3_27b.jsonl}"
TRIPLETS_OUTPUT="${TRIPLETS_OUTPUT:-data/triplets/triplets_27b_colbert.jsonl}"
MODEL="${MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
DOC_TOKENS="${DOC_TOKENS:-128}"
ENCODE_BATCH="${ENCODE_BATCH:-256}"
QUERY_BATCH="${QUERY_BATCH:-64}"
FIRST_STAGE="${FIRST_STAGE:-512}"
CANDIDATES="${CANDIDATES:-100}"
NEGATIVES="${NEGATIVES:-4}"
MIN_NEGATIVES="${MIN_NEGATIVES:-2}"
MAX_RATIO="${MAX_RATIO:-0.95}"
MIN_RATIO="${MIN_RATIO:-0.5}"
DUP_IDF="${DUP_IDF:-0.45}"
MAX_QUERIES="${MAX_QUERIES:-0}"
EXPORT="${EXPORT:-1}"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
[[ -z "$APPTAINER_BIN" ]] && { echo "ERROR: apptainer not available." >&2; exit 1; }
[[ ! -f "$IMAGE" ]] && { echo "ERROR: image not found: $IMAGE" >&2; exit 1; }

cd "$PROJ_ROOT"
mkdir -p logs "$(dirname "$TRIPLETS_OUTPUT")"

# The model is pulled from the Hub on first use; keep the cache in the project
# so a compute node without a writable home still finds it.
export APPTAINERENV_HF_HOME="${HF_HOME:-$PROJ_ROOT/.hf_cache}"
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${HF_HOME:-$PROJ_ROOT/.hf_cache}"

echo "=================================================================="
echo " Job        : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Model      : $MODEL"
echo " Output     : $TRIPLETS_OUTPUT"
echo " Score band : [$MIN_RATIO, $MAX_RATIO] x the positive's own MaxSim"
echo " Same-story : reclassify at idf containment >= $DUP_IDF (README §9)"
echo " Queries    : ${MAX_QUERIES:-all}"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$APPTAINER_BIN" exec --nv \
    --bind "$PROJ_ROOT":"$PROJ_ROOT" --pwd "$PROJ_ROOT" "$IMAGE" \
    python3 -m src.task2_triplets.build_triplets_colbert \
        --queries "$QUERIES" \
        --output "$TRIPLETS_OUTPUT" \
        --model "$MODEL" \
        --doc-max-tokens "$DOC_TOKENS" \
        --encode-batch "$ENCODE_BATCH" \
        --query-batch "$QUERY_BATCH" \
        --first-stage "$FIRST_STAGE" \
        --candidates "$CANDIDATES" \
        --negatives "$NEGATIVES" \
        --min-negatives "$MIN_NEGATIVES" \
        --max-neg-score-ratio "$MAX_RATIO" \
        --min-neg-score-ratio "$MIN_RATIO" \
        --dup-idf-containment "$DUP_IDF" \
        --max-queries "$MAX_QUERIES"

if [[ "$EXPORT" == "1" ]]; then
    HF_DIR="${HF_DIR:-data/triplets/hf}"
    HF_NAME="${HF_NAME:-$(basename "${TRIPLETS_OUTPUT%.jsonl}")}"
    HF_OUT="$HF_DIR/${HF_NAME}_hf.jsonl"
    mkdir -p "$HF_DIR"
    echo
    echo "▸ Exporting to $HF_OUT"
    "$APPTAINER_BIN" exec --bind "$PROJ_ROOT":"$PROJ_ROOT" --pwd "$PROJ_ROOT" "$IMAGE" \
        python3 -m src.task2_triplets.export_hf_dataset \
            --input "$TRIPLETS_OUTPUT" --output "$HF_OUT" ${SPLIT:+--split}

    PLOT_INPUTS=()
    if [[ -n "${SPLIT:-}" ]]; then
        for part in train eval test; do
            [[ -f "${HF_OUT%.jsonl}_${part}.jsonl" ]] && \
                PLOT_INPUTS+=(--input "${HF_OUT%.jsonl}_${part}.jsonl")
        done
    else
        PLOT_INPUTS=(--input "$HF_OUT")
    fi
    "$APPTAINER_BIN" exec --bind "$PROJ_ROOT":"$PROJ_ROOT" --pwd "$PROJ_ROOT" "$IMAGE" \
        python3 -m src.task2_triplets.plot_query_sources \
            "${PLOT_INPUTS[@]}" --output "$HF_DIR/${HF_NAME}_query_sources.png"
fi

echo "=================================================================="
echo " Done. Inspect with:"
echo "   python3 -m src.task2_triplets.inspect_triplets --input $TRIPLETS_OUTPUT"
echo "=================================================================="
