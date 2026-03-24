#!/bin/bash
# Activation script for the 3D Perception Pipeline project
# Usage: source activate.sh

TRT_LIB_DIR=/data/ganyw/3D/.venv/lib/python3.10/site-packages/tensorrt_libs
if [ -d "${TRT_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH=/data/ganyw/3D/.venv/lib:${TRT_LIB_DIR}:${LD_LIBRARY_PATH:-}
else
  export LD_LIBRARY_PATH=/data/ganyw/3D/.venv/lib:${LD_LIBRARY_PATH:-}
fi
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/data/ganyw/3D:/data/ganyw/3D/models/depth_anything_v2:${PYTHONPATH}
export GDINO_CKPT=/data/ganyw/3D/weights/groundingdino_swint_ogc.pth
export GDINO_CONFIG=/data/ganyw/3D/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
export SAM2_CKPT=/data/ganyw/3D/weights/sam2.1_hiera_small.pt
export DEPTH_V2_CKPT=/data/ganyw/3D/weights/depth_anything_v2_vits.pth
export CUDA_VISIBLE_DEVICES=0

source /data/ganyw/3D/.venv/bin/activate
echo "[3D Pipeline] Environment activated."
echo "  Python: $(python3 -c 'import sys; print(sys.executable)')"
echo "  PyTorch + CUDA: $(python3 -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
