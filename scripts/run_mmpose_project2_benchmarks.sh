#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/ganyw/3D"
VENV_PY="${ROOT}/.venv/bin/python"
MMPPOSE_DIR="${ROOT}/models/mmpose-main"
export LD_LIBRARY_PATH="${ROOT}/.venv/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${MMPPOSE_DIR}:${PYTHONPATH:-}"

CONFIG="${ROOT}/models/mmpose-main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
CHECKPOINT="${ROOT}/weights/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
IMAGE="${ROOT}/models/mmpose-main/projects/rtmpose/examples/onnxruntime/human-pose.jpeg"
ONNX_OUT="${ROOT}/exports/onnx/mmpose_hrnet_w48_256x192.onnx"
TRT_CACHE_DIR="${ROOT}/exports/tensorrt/mmpose_hrnet_w48_256x192"

mkdir -p "${ROOT}/benchmarks/mmpose" "${ROOT}/exports/onnx" "${ROOT}/exports/tensorrt"

if [ ! -f "${CHECKPOINT}" ]; then
  wget -O "${CHECKPOINT}" https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
fi

"${VENV_PY}" "${ROOT}/benchmarks/mmpose/benchmark_mmpose_pytorch.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --image "${IMAGE}" \
  --device cuda:0 \
  --warmup 20 \
  --runs 100 \
  --output "${ROOT}/benchmarks/mmpose/pytorch_fp32.json"

"${VENV_PY}" "${ROOT}/benchmarks/mmpose/export_mmpose_onnx.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --device cuda:0 \
  --output "${ONNX_OUT}"

"${VENV_PY}" "${ROOT}/benchmarks/mmpose/benchmark_mmpose_onnxruntime.py" \
  --config "${CONFIG}" \
  --onnx "${ONNX_OUT}" \
  --provider CUDAExecutionProvider \
  --image "${IMAGE}" \
  --warmup 20 \
  --runs 100 \
  --strict-provider \
  --output "${ROOT}/benchmarks/mmpose/onnxruntime_cuda.json"

if "${VENV_PY}" - <<'PY'
import onnxruntime as ort
print("TensorrtExecutionProvider" in ort.get_available_providers())
PY
then
  set +e
  "${VENV_PY}" "${ROOT}/benchmarks/mmpose/benchmark_mmpose_onnxruntime.py" \
    --config "${CONFIG}" \
    --onnx "${ONNX_OUT}" \
    --provider TensorrtExecutionProvider \
    --image "${IMAGE}" \
    --warmup 20 \
    --runs 100 \
    --strict-provider \
    --trt-fp16 \
    --trt-cache-dir "${TRT_CACHE_DIR}" \
    --output "${ROOT}/benchmarks/mmpose/onnxruntime_trt.json"
  set -e
fi

"${VENV_PY}" "${ROOT}/benchmarks/pipeline/generate_summary.py"
