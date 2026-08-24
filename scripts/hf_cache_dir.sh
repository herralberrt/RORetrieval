#!/usr/bin/env bash
#
# Choose an HF_HOME that can actually hold the model, and export it.
#
#   MODEL=google/gemma-3-27b-it source scripts/hf_cache_dir.sh
#
# Why this exists: the bf16 gemma-3-27b-it checkpoint is ~55 GiB of
# safetensors, while a student home on fep.grid.pub.ro is capped at 50 GiB
# (a hard CephFS quota - `getfattr -n ceph.quota.max_bytes ~`). Downloading
# there does not just fail, it fills the quota and breaks everything else on
# the way. When home is too small we fall back to node-local scratch, which
# is large (1.6 TB on the DGX nodes) but per-node and wiped between jobs, so
# the weights are re-fetched (~5-13 min at the ~70 MB/s a compute node gets).
#
# Set HF_HOME yourself to override; this script then only warns.

MODEL="${MODEL:-${GEMMA_MODEL:-google/gemma-3-27b-it}}"

# Approximate on-disk size of the bf16 weights, in GiB.
case "$MODEL" in
    *27b*) HF_MODEL_GB="${HF_MODEL_GB:-56}" ;;
    *12b*) HF_MODEL_GB="${HF_MODEL_GB:-25}" ;;
    *4b*)  HF_MODEL_GB="${HF_MODEL_GB:-9}"  ;;
    *1b*)  HF_MODEL_GB="${HF_MODEL_GB:-3}"  ;;
    *)     HF_MODEL_GB="${HF_MODEL_GB:-56}" ;;
esac
_need=$(( HF_MODEL_GB * 1024 * 1024 * 1024 ))

# Free bytes under the CephFS quota if there is one, else plain filesystem free.
_hf_free_bytes() {
    local quota used
    quota=$(getfattr --only-values -n ceph.quota.max_bytes "$HOME" 2>/dev/null || true)
    if [[ "$quota" =~ ^[0-9]+$ ]] && (( quota > 0 )); then
        used=$(getfattr --only-values -n ceph.dir.rbytes "$HOME" 2>/dev/null || echo 0)
        [[ "$used" =~ ^[0-9]+$ ]] || used=0
        echo $(( quota > used ? quota - used : 0 ))
    else
        df -B1 --output=avail "$HOME" 2>/dev/null | tail -1
    fi
}

_hf_cached() {   # already downloaded into $1?
    local dir="$1/hub/models--${MODEL//\//--}/snapshots"
    [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]
}

_gib() { echo $(( ${1:-0} / 1024 / 1024 / 1024 )); }

if [[ -n "${HF_HOME:-}" ]]; then
    echo "▸ HF cache: $HF_HOME (set explicitly)"
else
    _home_cache="$HOME/.cache/huggingface"
    _free=$(_hf_free_bytes)
    if _hf_cached "$_home_cache" || (( ${_free:-0} >= _need )); then
        export HF_HOME="$_home_cache"
        echo "▸ HF cache: $HF_HOME ($(_gib "$_free") GiB free, need ~${HF_MODEL_GB} GiB)"
    else
        export HF_HOME="${SLURM_TMPDIR:-/tmp}/$USER/hf"
        echo "▸ Home has only $(_gib "$_free") GiB free but $MODEL needs ~${HF_MODEL_GB} GiB."
        echo "▸ HF cache: $HF_HOME (node-local scratch; weights are re-fetched per node)"
    fi
fi

mkdir -p "$HF_HOME"

# `huggingface-cli login` stores the token at $HF_HOME/token, so relocating the
# cache hides it - and Gemma 3 is gated, so the download would 401 instead of
# failing on something obvious. Carry the token over as HF_TOKEN.
_home_token="$HOME/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && "$HF_HOME" != "$HOME/.cache/huggingface" && -s "$_home_token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$_home_token")"
    export HF_TOKEN
    echo "▸ Reusing the token from $_home_token (the relocated cache has none)"
fi

export HF_MODEL_GB
