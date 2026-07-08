#!/usr/bin/env python3
"""Export a trained RTDETRv2 checkpoint to ONNX and optionally TensorRT."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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


def main() -> None:
    args = parse_args()
    repo_pytorch_dir = resolve_repo_pytorch_dir(args.repo_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / f"{args.checkpoint.stem}.onnx"
    export_cmd = [
        sys.executable,
        "tools/export_onnx.py",
        "-c",
        str(args.config.resolve()),
        "-r",
        str(args.checkpoint.resolve()),
        "-o",
        str(onnx_path),
        "-s",
        str(args.input_size),
        "--check",
    ]
    if args.onnx_simplify:
        export_cmd.append("--simplify")
    run_command(export_cmd, cwd=repo_pytorch_dir)

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
