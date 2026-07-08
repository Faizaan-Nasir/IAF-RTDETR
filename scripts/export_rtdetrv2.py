#!/usr/bin/env python3
"""Export a trained RTDETRv2 checkpoint to ONNX and optionally TensorRT."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from rtdetrv2_utils import DEFAULT_REPO_DIR, resolve_repo_pytorch_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_REPO_DIR,
        help="Path to the cloned official RT-DETR repo root or its rtdetrv2_pytorch subdir.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the RTDETRv2 experiment config used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained .pth checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/fod_rtdetrv2/exports"),
        help="Directory to place exported artifacts in.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Export input size.",
    )
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="Only export ONNX; do not build a TensorRT engine.",
    )
    parser.add_argument(
        "--onnx-simplify",
        action="store_true",
        help="Simplify the exported ONNX model.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="TensorRT export max batch size.",
    )
    parser.add_argument(
        "--opt-batch-size",
        type=int,
        default=4,
        help="TensorRT export optimal batch size.",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="TensorRT export minimum batch size.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path) -> None:
    print(json.dumps({"cwd": str(cwd), "cmd": cmd}, indent=2))
    subprocess.run(cmd, cwd=cwd, check=True)


def export_onnx_model(
    repo_pytorch_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    input_size: int,
    simplify: bool,
) -> Path:
    sys.path.insert(0, str(repo_pytorch_dir))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().eval()
    images = torch.rand(1, 3, input_size, input_size)
    orig_target_sizes = torch.tensor([[input_size, input_size]])
    _ = model(images, orig_target_sizes)

    dynamic_axes = {
        "images": {0: "N"},
        "orig_target_sizes": {0: "N"},
    }

    torch.onnx.export(
        model,
        (images, orig_target_sizes),
        str(output_path),
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    if simplify:
        import onnxsim

        simplified_model, check = onnxsim.simplify(
            str(output_path),
            input_shapes={
                "images": images.shape,
                "orig_target_sizes": orig_target_sizes.shape,
            },
            dynamic_input_shape=True,
        )
        if not check:
            raise RuntimeError("onnxsim simplification reported failure")
        onnx.save(simplified_model, str(output_path))

    return output_path


def main() -> None:
    args = parse_args()
    repo_pytorch_dir = resolve_repo_pytorch_dir(args.repo_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / f"{args.checkpoint.stem}.onnx"
    export_onnx_model(
        repo_pytorch_dir=repo_pytorch_dir,
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        output_path=onnx_path,
        input_size=args.input_size,
        simplify=args.onnx_simplify,
    )

    summary = {
        "onnx": str(onnx_path),
    }

    if not args.skip_engine:
        engine_path = output_dir / f"{args.checkpoint.stem}.engine"
        trt_cmd = [
            sys.executable,
            "tools/export_trt.py",
            "-i",
            str(onnx_path),
            "-o",
            str(engine_path),
            "-Mb",
            str(args.max_batch_size),
            "-ob",
            str(args.opt_batch_size),
            "-mb",
            str(args.min_batch_size),
            "--fp16",
        ]
        run_command(trt_cmd, cwd=repo_pytorch_dir)
        summary["engine"] = str(engine_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
