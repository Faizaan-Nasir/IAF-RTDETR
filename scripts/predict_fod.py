#!/usr/bin/env python3
"""Run RTDETRv2 inference using the official deploy utilities."""

from __future__ import annotations

import argparse
import json
import shutil
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
        "--backend",
        choices=["torch", "onnxruntime", "tensorrt"],
        required=True,
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/fod_predict/result.jpg"),
        help="Output image path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Experiment config path required for torch backend.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required for torch backend.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="ONNX or TensorRT engine file required for those backends.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for torch or TensorRT inference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for TensorRT visualization.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path) -> None:
    print(json.dumps({"cwd": str(cwd), "cmd": cmd}, indent=2))
    subprocess.run(cmd, cwd=cwd, check=True)


def move_result(temp_dir: Path, output_path: Path) -> None:
    candidate = temp_dir / "results_0.jpg"
    if not candidate.exists():
        raise SystemExit(f"Expected inference image was not created: {candidate}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidate), str(output_path))


def main() -> None:
    args = parse_args()
    repo_pytorch_dir = resolve_repo_pytorch_dir(args.repo_dir)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "torch":
        if args.config is None or args.checkpoint is None:
            raise SystemExit("--config and --checkpoint are required for --backend torch")
        cmd = [
            sys.executable,
            str((repo_pytorch_dir / "references" / "deploy" / "rtdetrv2_torch.py").resolve()),
            "-c",
            str(args.config.resolve()),
            "-r",
            str(args.checkpoint.resolve()),
            "-f",
            str(args.source.resolve()),
            "-d",
            args.device,
        ]
        run_command(cmd, cwd=output_path.parent)
        move_result(output_path.parent, output_path)
    elif args.backend == "onnxruntime":
        if args.model is None:
            raise SystemExit("--model is required for --backend onnxruntime")
        cmd = [
            sys.executable,
            str((repo_pytorch_dir / "references" / "deploy" / "rtdetrv2_onnxruntime.py").resolve()),
            "--onnx-file",
            str(args.model.resolve()),
            "--im-file",
            str(args.source.resolve()),
        ]
        run_command(cmd, cwd=output_path.parent)
        move_result(output_path.parent, output_path)
    else:
        if args.model is None:
            raise SystemExit("--model is required for --backend tensorrt")
        cmd = [
            sys.executable,
            str((repo_pytorch_dir / "references" / "deploy" / "rtdetrv2_tensorrt.py").resolve()),
            "--engine",
            str(args.model.resolve()),
            "--image",
            str(args.source.resolve()),
            "--output",
            str(output_path),
            "--device",
            args.device,
            "--threshold",
            str(args.threshold),
        ]
        run_command(cmd, cwd=repo_pytorch_dir)

    print(
        json.dumps(
            {
                "backend": args.backend,
                "source": str(args.source.resolve()),
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
