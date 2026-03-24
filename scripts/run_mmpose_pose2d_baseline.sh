#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data/ganyw/3D"
VENV_PY="${ROOT_DIR}/.venv/bin/python"
MMPPOSE_DIR="${ROOT_DIR}/models/mmpose-main"
INPUT_IMAGE="${1:-tests/data/coco/000000197388.jpg}"
OUT_DIR="${2:-${ROOT_DIR}/visualizations/pose_cuda}"
DEVICE="${3:-cuda:0}"

export LD_LIBRARY_PATH="${ROOT_DIR}/.venv/lib:${LD_LIBRARY_PATH:-}"

cd "${MMPPOSE_DIR}"
"${VENV_PY}" demo/inferencer_demo.py "${INPUT_IMAGE}" \
  --pose2d configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
  --pose2d-weights https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
  --det-model whole_image \
  --device "${DEVICE}" \
  --vis-out-dir "${OUT_DIR}"
