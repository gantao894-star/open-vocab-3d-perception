#!/usr/bin/env python3
import argparse
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch


TRT_PROVIDER = "TensorrtExecutionProvider"


def preprocess(raw_bgr, input_size):
    img = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def summarize(lat_ms):
    arr = np.asarray(lat_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps": float(1000.0 / arr.mean()),
    }


def collect_cache_artifacts(cache_dir):
    if not cache_dir:
        return []
    patterns = ["*.engine", "*.profile", "*.timing", "*.json"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(cache_dir, pattern)))
    return sorted(set(files))


def make_provider_spec(args):
    if args.provider != TRT_PROVIDER:
        return args.provider

    options = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": args.trt_cache_dir,
        "trt_fp16_enable": args.trt_fp16,
    }
    if args.trt_max_workspace_gb is not None:
        options["trt_max_workspace_size"] = int(args.trt_max_workspace_gb * (1024**3))
    if args.trt_force_sequential_engine_build:
        options["trt_force_sequential_engine_build"] = True
    if args.trt_dump_subgraphs:
        options["trt_dump_subgraphs"] = True
    return (args.provider, options)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_failure(args, providers, error, active_providers=None):
    failure = {
        "backend": f"onnxruntime_{args.provider}",
        "status": "failed",
        "error": str(error),
        "providers_available": providers,
        "providers_active": active_providers or [],
        "requested_provider": args.provider,
        "trt_cache_dir": args.trt_cache_dir if args.provider == TRT_PROVIDER else None,
        "cache_artifacts": collect_cache_artifacts(args.trt_cache_dir) if args.provider == TRT_PROVIDER else [],
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    }
    write_json(args.output, failure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="/data/ganyw/3D/exports/onnx/depth_anything_v2_vits_518.onnx")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--image", default="/data/ganyw/3D/models/depth_anything_v2/assets/DA-2K.png")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", default="/data/ganyw/3D/benchmarks/depth/onnxruntime_cuda_vits_518.json")
    parser.add_argument("--strict-provider", action="store_true")
    parser.add_argument("--trt-cache-dir", default="/data/ganyw/3D/exports/tensorrt/depth_anything_v2_vits_518")
    parser.add_argument("--trt-fp16", action="store_true")
    parser.add_argument("--trt-max-workspace-gb", type=float, default=4.0)
    parser.add_argument("--trt-force-sequential-engine-build", action="store_true")
    parser.add_argument("--trt-dump-subgraphs", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    raw = cv2.imread(args.image)
    if raw is None:
        raise FileNotFoundError(f"Image not found or unreadable: {args.image}")
    inp = preprocess(raw, args.input_size)

    providers = ort.get_available_providers()
    if args.provider not in providers:
        raise RuntimeError(f"Provider {args.provider} not available. Available={providers}")

    provider_spec = make_provider_spec(args)
    provider_order = [provider_spec]
    for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
        if p != args.provider:
            provider_order.append(p)

    if args.provider == TRT_PROVIDER:
        os.makedirs(args.trt_cache_dir, exist_ok=True)

    try:
        t_init0 = time.perf_counter()
        session = ort.InferenceSession(args.onnx, providers=provider_order)
        init_ms = (time.perf_counter() - t_init0) * 1000.0
    except Exception as exc:
        write_failure(args, providers, exc)
        raise

    active_providers = session.get_providers()
    if args.strict_provider and args.provider not in active_providers:
        exc = RuntimeError(
            f"Requested provider {args.provider} not active. Active providers={active_providers}"
        )
        write_failure(args, providers, exc, active_providers)
        raise exc
    input_name = session.get_inputs()[0].name

    for _ in range(args.warmup):
        _ = session.run(None, {input_name: inp})

    lat_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: inp})
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    metrics = summarize(lat_ms)
    result = {
        "backend": f"onnxruntime_{args.provider}",
        "status": "ok",
        "input_size": args.input_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "providers_available": providers,
        "providers_active": active_providers,
        "session_init_ms": init_ms,
        "metrics": metrics,
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
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _ = torch.__version__
    main()
