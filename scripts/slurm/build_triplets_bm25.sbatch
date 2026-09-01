#!/bin/bash
#SBATCH --job-name=roq-bm25
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# TASK 2 (BM25 variant): mine lexical hard negatives for the generated queries.
#
# No GPU and no model download: BM25 is an inverted index in numpy, so this runs
# on the plain CPU partition (`haswell`) and leaves the DGX queue alone. The
# whole run - index 117k documents, score ~70k queries - is ~10 minutes; the
# 2-hour wall clock is slack for a full-corpus rerun with a wider candidate list.
#
# Memory is what to watch, not time: the corpus is held tokenised while each
# per-type index is built (~10 GB peak for `news`), hence 48G.
#
# Submit:
#     mkdir -p logs
#     sbatch --partition=haswell scripts/slurm/build_triplets_bm25.sbatch
#
# Override anything through the environment:
#     NEGATIVES=5 CANDIDATES=200 DUP_IDF=0.5 \
#         sbatch --partition=haswell scripts/slurm/build_triplets_bm25.sbatch
#
# A smoke run before the real one:
#     MAX_QUERIES=2000 TRIPLETS_OUTPUT=data/triplets/smoke_bm25.jsonl \
#         sbatch --partition=haswell scripts/slurm/build_triplets_bm25.sbatch

set -euo pipefail

PROJ_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJ_ROOT/containers/roretrieval.sif}"

QUERIES="${QUERIES:-data/queries/queries_gemma3_27b.jsonl}"
TRIPLETS_OUTPUT="${TRIPLETS_OUTPUT:-data/triplets/triplets_27b_bm25.jsonl}"
NEGATIVES="${NEGATIVES:-4}"
MIN_NEGATIVES="${MIN_NEGATIVES:-2}"
CANDIDATES="${CANDIDATES:-100}"
MAX_NEG_RATIO="${MAX_NEG_RATIO:-0.6}"
MIN_NEG_RATIO="${MIN_NEG_RATIO:-0.15}"
DUP_OVERLAP="${DUP_OVERLAP:-0.7}"
DUP_TITLE="${DUP_TITLE:-0.6}"
DUP_IDF="${DUP_IDF:-0.45}"           # candidate vs. the POSITIVE DOCUMENT
TITLE_PREFIX="${TITLE_PREFIX:-5}"
MIN_LEX_TITLE="${MIN_LEX_TITLE:-0.2}"
MIN_TITLE_TERMS="${MIN_TITLE_TERMS:-3}"
MIN_DOC_CHARS="${MIN_DOC_CHARS:-300}"
IDF_COVERAGE="${IDF_COVERAGE:-0.3}"
MAX_POS_RANK="${MAX_POS_RANK:-5}"
MAX_QUERIES="${MAX_QUERIES:-0}"      # 0 = all

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"

if [[ -z "$APPTAINER_BIN" ]]; then
    echo "ERROR: apptainer/singularity not available on this node." >&2
    exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: image not found: $IMAGE  (build it with containers/build.sh)" >&2
    exit 1
fi

cd "$PROJ_ROOT"
mkdir -p logs "$(dirname "$TRIPLETS_OUTPUT")"

export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=================================================================="
echo " Job        : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Queries    : $QUERIES"
echo " Output     : $TRIPLETS_OUTPUT"
echo " Negatives  : $NEGATIVES (at least $MIN_NEGATIVES) from $CANDIDATES BM25 candidates"
echo " Score band : [$MIN_NEG_RATIO, $MAX_NEG_RATIO] x the positive's own score"
echo " Same story : reclassify as POSITIVE any candidate sharing >= $DUP_IDF of"
echo "              the positive's idf mass, >= $DUP_OVERLAP of its terms, or"
echo "              >= $DUP_TITLE of its title stems (${TITLE_PREFIX}-char prefixes)"
echo "=================================================================="

"$APPTAINER_BIN" exec \
    --bind "$PROJ_ROOT":"$PROJ_ROOT" \
    --pwd "$PROJ_ROOT" \
    "$IMAGE" \
    python3 -m src.task2_triplets.build_triplets_bm25 \
        --queries "$QUERIES" \
        --output "$TRIPLETS_OUTPUT" \
        --negatives "$NEGATIVES" \
        --min-negatives "$MIN_NEGATIVES" \
        --candidates "$CANDIDATES" \
        --max-neg-score-ratio "$MAX_NEG_RATIO" \
        --min-neg-score-ratio "$MIN_NEG_RATIO" \
        --dup-token-overlap "$DUP_OVERLAP" \
        --dup-title-overlap "$DUP_TITLE" \
        --dup-idf-containment "$DUP_IDF" \
        --title-prefix "$TITLE_PREFIX" \
        --min-lex-for-title "$MIN_LEX_TITLE" \
        --min-title-terms "$MIN_TITLE_TERMS" \
        --min-doc-chars "$MIN_DOC_CHARS" \
        --min-query-idf-coverage "$IDF_COVERAGE" \
        --max-positive-rank "$MAX_POS_RANK" \
        --max-queries "$MAX_QUERIES"

# ------------------------------------------------------------------ export
# The training layout (alina0195/ro-msmarco-divided columns + query_source) and
# the source plot, in the same job: they read the file that was just written, so
# splitting them into a second submission only adds a queue wait and a chance to
# forget. Set EXPORT=0 to skip.
if [[ "${EXPORT:-1}" == "1" ]]; then
    HF_DIR="${HF_DIR:-data/triplets/hf}"
    HF_NAME="${HF_NAME:-$(basename "${TRIPLETS_OUTPUT%.jsonl}")}"
    HF_OUT="$HF_DIR/${HF_NAME}_hf.jsonl"
    mkdir -p "$HF_DIR"

    echo
    echo "▸ Exporting to $HF_OUT"
    "$APPTAINER_BIN" exec \
        --bind "$PROJ_ROOT":"$PROJ_ROOT" --pwd "$PROJ_ROOT" "$IMAGE" \
        python3 -m src.task2_triplets.export_hf_dataset \
            --input "$TRIPLETS_OUTPUT" \
            --output "$HF_OUT" \
            ${SPLIT:+--split}

    # With --split the exporter writes <stem>_train/_eval/_test and no base
    # file, so the plot has to be pointed at the parts, not at $HF_OUT.
    PLOT_INPUTS=()
    if [[ -n "${SPLIT:-}" ]]; then
        for part in train eval test; do
            [[ -f "${HF_OUT%.jsonl}_${part}.jsonl" ]] && \
                PLOT_INPUTS+=(--input "${HF_OUT%.jsonl}_${part}.jsonl")
        done
    else
        PLOT_INPUTS=(--input "$HF_OUT")
    fi

    echo
    echo "▸ Plotting query_source"
    "$APPTAINER_BIN" exec \
        --bind "$PROJ_ROOT":"$PROJ_ROOT" --pwd "$PROJ_ROOT" "$IMAGE" \
        python3 -m src.task2_triplets.plot_query_sources \
            "${PLOT_INPUTS[@]}" \
            --output "$HF_DIR/${HF_NAME}_query_sources.png"
fi

echo "=================================================================="
echo " Done. Inspect the triplets with:"
echo "   python3 -m src.task2_triplets.inspect_triplets --input $TRIPLETS_OUTPUT"
echo
echo " Then push (needs a token; --yes is required to actually upload):"
echo "   HF_TOKEN=hf_... apptainer exec containers/roretrieval.sif \\"
echo "     python3 -m src.task2_triplets.push_to_hf \\"
echo "       --input ${HF_OUT:-data/triplets/hf/..._hf.jsonl} \\"
echo "       --repo PaulBurca2005/ro-retrieval-triplets --public --yes"
echo "=================================================================="
