#!/usr/bin/env python3
import argparse
import glob
import os
import time

import onnxruntime as ort
import torch

from common import infer_input_size, load_cfg, preprocess_image, summarize, write_json

TRT_PROVIDER = "TensorrtExecutionProvider"


def collect_cache_artifacts(cache_dir):
    if not cache_dir:
        return []
    files = []
    for pattern in ["*.engine", "*.profile", "*.timing", "*.json"]:
        files.extend(glob.glob(os.path.join(cache_dir, pattern)))
    return sorted(set(files))


def write_failure(args, providers, input_size, error, active_providers=None):
    failure = {
        "backend": f"onnxruntime_{args.provider}",
        "status": "failed",
        "config": args.config,
        "onnx": args.onnx,
        "input_size": {"width": input_size[0], "height": input_size[1]},
        "error": str(error),
        "providers_available": providers,
        "providers_active": active_providers or [],
        "cache_artifacts": collect_cache_artifacts(args.trt_cache_dir) if args.provider == TRT_PROVIDER else [],
    }
    write_json(args.output, failure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--image", default="/data/ganyw/3D/models/mmpose-main/projects/rtmpose/examples/onnxruntime/human-pose.jpeg")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict-provider", action="store_true")
    parser.add_argument("--trt-cache-dir", default="/data/ganyw/3D/exports/tensorrt/mmpose")
    parser.add_argument("--trt-fp16", action="store_true")
    parser.add_argument("--trt-max-workspace-gb", type=float, default=4.0)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    input_size = infer_input_size(cfg)
    inp = preprocess_image(args.image, input_size)

    providers = ort.get_available_providers()
    if args.provider not in providers:
        raise RuntimeError(f"Provider {args.provider} not available. Available={providers}")

    if args.provider == TRT_PROVIDER:
        os.makedirs(args.trt_cache_dir, exist_ok=True)
        provider_spec = (
            args.provider,
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": args.trt_cache_dir,
                "trt_fp16_enable": args.trt_fp16,
                "trt_max_workspace_size": int(args.trt_max_workspace_gb * (1024**3)),
            },
        )
    else:
        provider_spec = args.provider

    provider_order = [provider_spec]
    for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
        if p != args.provider:
            provider_order.append(p)

    try:
        t0 = time.perf_counter()
        session = ort.InferenceSession(args.onnx, providers=provider_order)
        init_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        write_failure(args, providers, input_size, exc)
        raise

    active_providers = session.get_providers()
    if args.strict_provider and args.provider not in active_providers:
        exc = RuntimeError(f"Requested provider {args.provider} not active. Active providers={active_providers}")
        write_failure(args, providers, input_size, exc, active_providers)
        raise exc

    input_name = session.get_inputs()[0].name
    output_names = [x.name for x in session.get_outputs()]

    for _ in range(args.warmup):
        session.run(output_names, {input_name: inp})

    lat_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        session.run(output_names, {input_name: inp})
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "backend": f"onnxruntime_{args.provider}",
        "status": "ok",
        "config": args.config,
        "onnx": args.onnx,
        "input_size": {"width": input_size[0], "height": input_size[1]},
        "warmup": args.warmup,
        "runs": args.runs,
        "providers_available": providers,
        "providers_active": active_providers,
        "session_init_ms": init_ms,
        "metrics": summarize(lat_ms),
    }
    if args.provider == TRT_PROVIDER:
        result.update(
            {
                "trt_cache_dir": args.trt_cache_dir,
                "trt_fp16": args.trt_fp16,
                "trt_max_workspace_gb": args.trt_max_workspace_gb,
                "cache_artifacts": collect_cache_artifacts(args.trt_cache_dir),
            }
        )
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    _ = torch.__version__
    main()
