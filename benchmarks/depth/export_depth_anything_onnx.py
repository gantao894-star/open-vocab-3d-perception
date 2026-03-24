#!/usr/bin/env python3
import argparse
import os

import torch

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/data/ganyw/3D/models/depth_anything_v2")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--output", default="/data/ganyw/3D/exports/onnx/depth_anything_v2_vits_518.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt = os.path.join(args.repo_root, "checkpoints", f"depth_anything_v2_{args.encoder}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.eval().cpu()

    dummy = torch.randn(1, 3, args.input_size, args.input_size, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["depth"],
            dynamic_axes={"input": {0: "batch"}, "depth": {0: "batch"}},
        )

    print(f"ONNX exported: {args.output}")


if __name__ == "__main__":
    main()
