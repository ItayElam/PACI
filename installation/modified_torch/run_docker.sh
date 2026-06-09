#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-modified-torch-build:2.4.0}"
CONTAINER_WORKSPACE="/workspace"

echo "Building image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"

RUN_ARGS=(
  --rm
  -it
  -v "${SCRIPT_DIR}:${CONTAINER_WORKSPACE}"
  -w "${CONTAINER_WORKSPACE}"
  --gpus all
)

echo "Starting container (host ${SCRIPT_DIR} -> ${CONTAINER_WORKSPACE})"
echo "Wheels written under ${CONTAINER_WORKSPACE} appear in this directory on the host."

exec docker run "${RUN_ARGS[@]}" "${IMAGE_NAME}"
