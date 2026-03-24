# Benchmark Summary

Generated from JSON artifacts under `benchmarks/`.

## Depth Anything V2 (ViT-S, 518x518)
| Backend | Status | Mean ms | FPS | P95 ms | Notes | Source |
|---|---:|---:|---:|---:|---|---|
| pytorch_fp32 | ok | 51.68 | 19.35 | 52.57 | - | `benchmarks/depth/pytorch_fp32_vits_518_cuda.json` |
| onnxruntime_CUDAExecutionProvider | ok | 47.43 | 21.08 | 48.39 | active=CUDAExecutionProvider,CPUExecutionProvider | `benchmarks/depth/onnxruntime_cuda_vits_518.json` |
| onnxruntime_TensorrtExecutionProvider | failed | - | - | - | active=CPUExecutionProvider; Requested provider TensorrtExecutionProvider not active. Active providers=['CPUExecutionProvider'] | `benchmarks/depth/onnxruntime_trt_ep_vits_518.json` |

## MMPose 2D Pose
No benchmark artifacts found.

## Blockers
- TensorRT engine generation on this machine is currently blocked if `libnvinfer.so.10` is missing from the runtime library path.
- ONNX Runtime may advertise `TensorrtExecutionProvider`, but provider activation still fails until TensorRT runtime libraries are installed and discoverable.

