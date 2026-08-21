#!/usr/bin/env bash
#
# Build the RORetrieval Apptainer image.
#
#   bash containers/build.sh              # build to containers/roretrieval.sif
#   IMAGE=/scratch/me/ro.sif bash containers/build.sh
#   BUILD_MODE=remote bash containers/build.sh   # use the Sylabs remote builder
#
# Building needs root or fakeroot. On a cluster login node without either,
# use BUILD_MODE=remote (requires `apptainer remote login`), or build on your
# own machine and copy the .sif over with scp/rsync.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${DEF_FILE:-$SCRIPT_DIR/roretrieval.def}"
IMAGE="${IMAGE:-$SCRIPT_DIR/roretrieval.sif}"
BUILD_MODE="${BUILD_MODE:-auto}"

if ! command -v apptainer >/dev/null 2>&1; then
    if command -v singularity >/dev/null 2>&1; then
        APPTAINER=singularity
    else
        echo "ERROR: neither apptainer nor singularity found in PATH." >&2
        echo "       On a cluster try: module load apptainer   (or module avail apptainer)" >&2
        exit 1
    fi
else
    APPTAINER=apptainer
fi

if [[ ! -f "$DEF_FILE" ]]; then
    echo "ERROR: definition file not found: $DEF_FILE" >&2
    exit 1
fi

if [[ -e "$IMAGE" ]]; then
    echo "Image already exists: $IMAGE"
    read -r -p "Overwrite it? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    BUILD_ARGS=(--force)
else
    BUILD_ARGS=()
fi

case "$BUILD_MODE" in
    remote)  BUILD_ARGS+=(--remote) ;;
    fakeroot) BUILD_ARGS+=(--fakeroot) ;;
    sudo)    ;;
    auto)
        if [[ "$(id -u)" -ne 0 ]]; then
            BUILD_ARGS+=(--fakeroot)
        fi
        ;;
    *)
        echo "ERROR: unknown BUILD_MODE=$BUILD_MODE (auto|fakeroot|sudo|remote)" >&2
        exit 1
        ;;
esac

echo "==> Building $IMAGE"
echo "    definition : $DEF_FILE"
echo "    mode       : $BUILD_MODE"
echo "    command    : $APPTAINER build ${BUILD_ARGS[*]-} $IMAGE $DEF_FILE"
echo "    (this pulls ~10GB of CUDA + torch + vLLM; expect 15-30 minutes)"
echo

if [[ "$BUILD_MODE" == "sudo" ]]; then
    sudo "$APPTAINER" build ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} "$IMAGE" "$DEF_FILE"
else
    "$APPTAINER" build ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} "$IMAGE" "$DEF_FILE"
fi

echo
echo "==> Done: $IMAGE"
echo "    Smoke test (needs a GPU node):"
echo "      $APPTAINER exec --nv $IMAGE python3 -c 'import torch; print(torch.cuda.is_available())'"
