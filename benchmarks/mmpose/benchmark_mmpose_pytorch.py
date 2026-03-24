#!/usr/bin/env python3
import argparse
import time

import torch

from common import infer_input_size, init_pose_model, load_cfg, preprocess_image, summarize, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default="/data/ganyw/3D/models/mmpose-main/projects/rtmpose/examples/onnxruntime/human-pose.jpeg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    input_size = infer_input_size(cfg)
    inp_np = preprocess_image(args.image, input_size)

    model = init_pose_model(args.config, args.checkpoint, args.device)
    inp = torch.from_numpy(inp_np).to(args.device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(inp, None, mode="tensor")
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        lat_ms = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            _ = model(inp, None, mode="tensor")
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "backend": "pytorch_fp32",
        "status": "ok",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "input_size": {"width": input_size[0], "height": input_size[1]},
        "device": args.device,
        "warmup": args.warmup,
        "runs": args.runs,
        "metrics": summarize(lat_ms),
    }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
