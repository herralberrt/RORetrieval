#!/bin/bash
#SBATCH --job-name=ro-download
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Download the corpus into data/categories/ from inside the container.
#
# This runs as a batch job rather than on the login node on purpose: Apptainer
# mounts the .sif through squashfuse, and that mount is torn down when the
# session that started it ends ("Terminating squashfuse_ll after timeout"),
# killing any long download detached with nohup. A batch job owns its session
# for its whole lifetime.
#
# Compute nodes on this cluster do have outbound internet, which the download
# needs (HuggingFace + a GitHub clone).
#
#     mkdir -p logs
#     sbatch --partition=haswell scripts/slurm/download_data.sbatch
#     DATASET=news sbatch --partition=haswell scripts/slurm/download_data.sbatch
#
# No GPU is needed, so use a CPU partition and leave the GPU queues free.

set -euo pipefail

PROJ_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJ_ROOT/containers/roretrieval.sif}"
DATASET="${DATASET:-all}"          # all | recipes | stories | summarization | news
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
APPTAINER_BIN="$(command -v apptainer || command -v singularity)"

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: image not found: $IMAGE" >&2
    exit 1
fi

cd "$PROJ_ROOT"
mkdir -p logs data/categories "$HF_HOME"

export APPTAINERENV_HF_HOME="$HF_HOME"
# The news downloader clones ~350MB of JSON; the default 2 minute timeout in
# the script is not enough on a shared link.
export APPTAINERENV_NEWS_CLONE_TIMEOUT="${NEWS_CLONE_TIMEOUT:-1800}"

echo "=================================================================="
echo " Job     : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Dataset : $DATASET"
echo " Target  : $PROJ_ROOT/data/categories"
echo "=================================================================="

ARGS=()
[[ "$DATASET" != "all" ]] && ARGS=(--dataset "$DATASET")

"$APPTAINER_BIN" exec \
    --bind "$PROJ_ROOT":"$PROJ_ROOT" \
    --bind "$HF_HOME":"$HF_HOME" \
    --pwd "$PROJ_ROOT" \
    "$IMAGE" \
    python3 -m src.data_prep.download_datasets ${ARGS[@]+"${ARGS[@]}"}

echo "=================================================================="
du -sh data/categories
find data/categories -name "*.jsonl" -size +1k -printf "  %s\t%p\n" | sort -rn
echo "=================================================================="
