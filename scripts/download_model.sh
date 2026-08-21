#!/usr/bin/env bash
#
# Pre-download the Gemma weights into HF_HOME from a node that has internet
# (usually the login node). Compute nodes are often offline, in which case the
# generation job must find the weights already cached.
#
#   source containers/env.sh        # provides HF_TOKEN / HF_HOME
#   bash scripts/download_model.sh
#   bash scripts/download_model.sh google/gemma-3-12b-it

set -euo pipefail

MODEL="${1:-${GEMMA_MODEL:-google/gemma-3-4b-it}}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
IMAGE="${IMAGE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/containers/roretrieval.sif}"

export HF_HOME
mkdir -p "$HF_HOME"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2
    echo "       Gemma 3 is gated: accept the license at" >&2
    echo "       https://huggingface.co/${MODEL} and create a read token." >&2
    exit 1
fi

echo "==> Downloading $MODEL into $HF_HOME"

if command -v apptainer >/dev/null 2>&1 && [[ -f "$IMAGE" ]]; then
    APPTAINERENV_HF_HOME="$HF_HOME" APPTAINERENV_HF_TOKEN="$HF_TOKEN" \
    apptainer exec --bind "$HF_HOME":"$HF_HOME" "$IMAGE" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','*.model','*.txt'])
print('cached')
"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','*.model','*.txt'])
print('cached')
"
fi

echo "==> Done. Compute nodes can now run with HF_HUB_OFFLINE=1."
