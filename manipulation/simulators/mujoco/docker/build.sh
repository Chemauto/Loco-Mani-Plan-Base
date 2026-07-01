#!/bin/bash
set -e
docker build -t loco-mujoco-ros:latest "$(dirname "$0")"
echo "镜像构建完成: loco-mujoco-ros:latest"
