#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/ganyw/3D"
VENV_PY="${ROOT}/.venv/bin/python"
DEPTH_REPO="${ROOT}/models/depth_anything_v2"

export LD_LIBRARY_PATH="${ROOT}/.venv/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${DEPTH_REPO}:${PYTHONPATH:-}"

mkdir -p "${ROOT}/benchmarks/depth" "${ROOT}/exports/onnx"
rm -f "${ROOT}/benchmarks/depth/onnxruntime_trt_ep_vits_518.json"

"${VENV_PY}" "${ROOT}/benchmarks/depth/benchmark_depth_pytorch.py" \
  --repo-root "${DEPTH_REPO}" \
  --encoder vits \
  --input-size 518 \
  --device cuda:0 \
  --warmup 20 \
  --runs 100 \
  --output "${ROOT}/benchmarks/depth/pytorch_fp32_vits_518_cuda.json"

"${VENV_PY}" "${ROOT}/benchmarks/depth/export_depth_anything_onnx.py" \
  --repo-root "${DEPTH_REPO}" \
  --encoder vits \
  --input-size 518 \
  --output "${ROOT}/exports/onnx/depth_anything_v2_vits_518.onnx"

"${VENV_PY}" "${ROOT}/benchmarks/depth/benchmark_depth_onnxruntime.py" \
  --onnx "${ROOT}/exports/onnx/depth_anything_v2_vits_518.onnx" \
  --provider CUDAExecutionProvider \
  --input-size 518 \
  --warmup 20 \
  --runs 100 \
  --strict-provider \
  --output "${ROOT}/benchmarks/depth/onnxruntime_cuda_vits_518.json"

if "${VENV_PY}" - <<'PY'
import onnxruntime as ort
print("TensorrtExecutionProvider" in ort.get_available_providers())
PY
then
  "${VENV_PY}" "${ROOT}/benchmarks/depth/benchmark_depth_onnxruntime.py" \
    --onnx "${ROOT}/exports/onnx/depth_anything_v2_vits_518.onnx" \
    --provider TensorrtExecutionProvider \
    --input-size 518 \
    --warmup 20 \
    --runs 100 \
    --strict-provider \
    --trt-fp16 \
    --trt-cache-dir "${ROOT}/exports/tensorrt/depth_anything_v2_vits_518" \
    --output "${ROOT}/benchmarks/depth/onnxruntime_trt_ep_vits_518.json" \
  || echo "TensorRT EP benchmark skipped: provider not active."
fi

"${VENV_PY}" - <<'PY'
import json
from pathlib import Path

root = Path("/data/ganyw/3D/benchmarks/depth")
paths = [
    root / "pytorch_fp32_vits_518_cuda.json",
    root / "onnxruntime_cuda_vits_518.json",
    root / "onnxruntime_trt_ep_vits_518.json",
]

rows = []
for p in paths:
    if p.exists():
        data = json.loads(p.read_text())
        if data["backend"].endswith("TensorrtExecutionProvider"):
            active = data.get("providers_active", [])
            if "TensorrtExecutionProvider" not in active:
                continue
        rows.append((data["backend"], data["metrics"]["mean_ms"], data["metrics"]["fps"], data["metrics"]["p95_ms"]))

print("backend,mean_ms,fps,p95_ms")
for r in rows:
    print(f"{r[0]},{r[1]:.3f},{r[2]:.2f},{r[3]:.3f}")
PY

"${VENV_PY}" "${ROOT}/benchmarks/pipeline/generate_summary.py"
