#!/usr/bin/env python3
"""Train RTDETRv2 on the single-class FOD dataset using the official repo."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rtdetrv2_utils import DEFAULT_DATASET_NAME, DEFAULT_REPO_DIR, find_checkpoint, prepare_rtdetrv2_workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_REPO_DIR,
        help="Path to the cloned official RT-DETR repo root or its rtdetrv2_pytorch subdir.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fod_single_class/coco"),
        help="Path to the prepared COCO-format dataset root.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset link/config name to create inside the RTDETRv2 repo.",
    )
    parser.add_argument(
        "--variant",
        choices=["l", "s"],
        default="l",
        help="RTDETRv2 model size to scaffold.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers to write into the generated config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run/config name. Defaults to fod_rtdetrv2_<variant>_<epochs>e.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/fod_rtdetrv2"),
        help="Base directory for training outputs.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use. Set >1 to launch with torchrun.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed.",
    )
    parser.add_argument(
        "--tuning-checkpoint",
        type=Path,
        default=None,
        help="Optional RTDETRv2 checkpoint to fine-tune from.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to resume training from.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device passed to the official training script.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP mixed-precision training.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Only train; do not auto-export ONNX/TensorRT afterward.",
    )
    parser.add_argument(
        "--onnx-simplify",
        action="store_true",
        help="Simplify the exported ONNX graph.",
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
    run_name = args.run_name or f"fod_rtdetrv2_{args.variant}_{args.epochs}e"
    output_dir = (args.output_dir / run_name).resolve()

    workspace = prepare_rtdetrv2_workspace(
        repo_dir=args.repo_dir,
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        variant=args.variant,
        epochs=args.epochs,
        run_name=run_name,
        workers=args.workers,
    )
    repo_pytorch_dir = workspace["repo_pytorch_dir"]
    experiment_config = workspace["experiment_config"]

    if args.gpus > 1:
        train_cmd = [
            "torchrun",
            f"--nproc_per_node={args.gpus}",
            "tools/train.py",
        ]
    else:
        train_cmd = [sys.executable, "tools/train.py"]

    train_cmd.extend(
        [
            "-c",
            str(experiment_config),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--output-dir",
            str(output_dir),
        ]
    )

    if not args.no_amp:
        train_cmd.append("--use-amp")
    if args.tuning_checkpoint is not None:
        train_cmd.extend(["-t", str(args.tuning_checkpoint.resolve())])
    if args.resume_checkpoint is not None:
        train_cmd.extend(["-r", str(args.resume_checkpoint.resolve())])

    run_command(train_cmd, cwd=repo_pytorch_dir)

    checkpoint = find_checkpoint(output_dir)
    summary = {
        "repo_pytorch_dir": str(repo_pytorch_dir),
        "experiment_config": str(experiment_config),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
    }

    if not args.skip_export:
        export_cmd = [
            sys.executable,
            str((Path.cwd() / "scripts" / "export_rtdetrv2.py").resolve()),
            "--repo-dir",
            str(repo_pytorch_dir),
            "--config",
            str(experiment_config),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir / "exports"),
            "--max-batch-size",
            str(args.max_batch_size),
            "--opt-batch-size",
            str(args.opt_batch_size),
            "--min-batch-size",
            str(args.min_batch_size),
        ]
        if args.onnx_simplify:
            export_cmd.append("--onnx-simplify")
        run_command(export_cmd, cwd=Path.cwd())
        summary["exports_dir"] = str((output_dir / "exports").resolve())

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
