#!/bin/bash
set -e
WS_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
MUJOCO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

xhost +local: > /dev/null 2>&1

docker run -it --rm \
  --network host \
  --ipc host \
  -e DISPLAY="${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/.Xauthority:/root/.Xauthority:ro" \
  -v "${WS_ROOT}:${WS_ROOT}" \
  -w "${MUJOCO_DIR}" \
  loco-mujoco-ros:latest \
  /bin/bash
