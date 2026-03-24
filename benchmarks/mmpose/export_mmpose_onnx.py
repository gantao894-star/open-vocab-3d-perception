#!/usr/bin/env python3
import argparse
import os

import torch

from common import TensorModeWrapper, infer_input_size, init_pose_model, load_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    input_size = infer_input_size(cfg)
    model = init_pose_model(args.config, args.checkpoint, args.device)
    wrapper = TensorModeWrapper(model).to(args.device).eval()
    dummy = torch.randn(1, 3, input_size[1], input_size[0], dtype=torch.float32, device=args.device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output_0"],
            dynamic_axes={"input": {0: "batch"}, "output_0": {0: "batch"}},
        )
    print(f"ONNX exported: {args.output}")


if __name__ == "__main__":
    main()
