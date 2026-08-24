#!/usr/bin/env bash
#
# Pre-download the Gemma weights into HF_HOME from a node that has internet
# (usually the login node). Compute nodes are often offline, in which case the
# generation job must find the weights already cached.
#
#   source containers/env.sh        # provides HF_TOKEN / HF_HOME
#   bash scripts/download_model.sh
#   bash scripts/download_model.sh google/gemma-3-12b-it
#
# NOTE for fep.grid.pub.ro: the bf16 gemma-3-27b-it checkpoint is ~55 GiB and a
# student home is capped at 50 GiB, so it cannot be cached there. The compute
# nodes on dgxa100/dgxh100 do have outbound internet, so the SLURM job fetches
# the weights into node-local scratch instead - you do not need this script for
# the 27B model unless you point HF_HOME at storage large enough to hold it.

set -euo pipefail

MODEL="${1:-${GEMMA_MODEL:-google/gemma-3-27b-it}}"
IMAGE="${IMAGE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/containers/roretrieval.sif}"

# Chooses (and exports) an HF_HOME with room for the weights.
source "$(dirname "${BASH_SOURCE[0]}")/hf_cache_dir.sh"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2
    echo "       Gemma 3 is gated: accept the license at" >&2
    echo "       https://huggingface.co/${MODEL} and create a read token." >&2
    echo "       License acceptance is per repository - accepting it for" >&2
    echo "       gemma-3-4b-it does not cover gemma-3-27b-it." >&2
    exit 1
fi

echo "==> Downloading $MODEL into $HF_HOME (~${HF_MODEL_GB} GiB)"

# HF_HUB_ENABLE_HF_TRANSFER is read when huggingface_hub is imported, so it has
# to be in the environment already - setting it inside the script is too late.
DOWNLOAD_PY="
from huggingface_hub import snapshot_download
snapshot_download(
    '$MODEL',
    allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'],
    max_workers=8,
)
print('cached')
"

if command -v apptainer >/dev/null 2>&1 && [[ -f "$IMAGE" ]]; then
    APPTAINERENV_HF_HOME="$HF_HOME" APPTAINERENV_HF_TOKEN="$HF_TOKEN" \
    APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=1 \
    apptainer exec --bind "$HF_HOME":"$HF_HOME" "$IMAGE" python3 -c "$DOWNLOAD_PY"
else
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "$DOWNLOAD_PY"
fi

echo "==> Done. Compute nodes can now run with HF_HUB_OFFLINE=1."
