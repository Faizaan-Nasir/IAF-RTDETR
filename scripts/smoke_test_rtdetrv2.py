#!/usr/bin/env python3
"""Run a CPU smoke test for the official RTDETRv2 pipeline on the prepared FOD dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from PIL import Image, ImageDraw
import torchvision.transforms as T

from export_rtdetrv2 import export_onnx_model
from rtdetrv2_utils import DEFAULT_DATASET_NAME, DEFAULT_REPO_DIR, prepare_rtdetrv2_workspace


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
        help="RTDETRv2 model size to validate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Reference epoch count for generated configs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Reference worker count for generated configs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run/config name. Defaults to fod_rtdetrv2_<variant>_<epochs>e_smoke.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/fod_rtdetrv2_smoke"),
        help="Directory to place smoke-test outputs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the smoke test. Defaults to CPU for portability.",
    )
    return parser.parse_args()


def write_smoke_config(experiment_config: Path, run_name: str) -> Path:
    smoke_config_path = experiment_config.with_name(f"{run_name}_smoke.yml")
    config = yaml.safe_load(experiment_config.read_text())

    config["output_dir"] = f"./output/{run_name}_smoke"
    config["epoches"] = 1
    config.setdefault("PResNet", {})
    config["PResNet"]["pretrained"] = False

    for split_name in ("train_dataloader", "val_dataloader"):
        config.setdefault(split_name, {})
        config[split_name]["num_workers"] = 0
        config[split_name]["total_batch_size"] = 1

    config.setdefault("train_dataloader", {}).setdefault("collate_fn", {})
    config["train_dataloader"]["collate_fn"]["scales"] = [640]
    config["train_dataloader"]["collate_fn"]["stop_epoch"] = 0
    config.setdefault("train_dataloader", {}).setdefault("dataset", {}).setdefault("transforms", {}).setdefault(
        "policy", {}
    )
    config["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] = 0

    smoke_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return smoke_config_path


def first_image(path: Path) -> Path:
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        matches = sorted(path.glob(suffix))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No image files found in {path}")


def draw_onnx_predictions(
    image_path: Path,
    onnx_path: Path,
    output_path: Path,
    threshold: float = 0.25,
) -> None:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    image_tensor = transforms(image)[None]
    orig_target_sizes = np.array([[width, height]], dtype=np.int64)
    labels, boxes, scores = session.run(
        output_names=None,
        input_feed={
            "images": image_tensor.numpy(),
            "orig_target_sizes": orig_target_sizes,
        },
    )

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    kept = 0
    for label, box, score in zip(labels[0], boxes[0], scores[0]):
        if float(score) < threshold:
            continue
        kept += 1
        x0, y0, x1, y1 = [float(value) for value in box]
        draw.rectangle(((x0, y0), (x1, y1)), outline="red", width=2)
        draw.text((x0, y0), f"FOD {float(score):.2f}", fill="red")

    annotated.save(output_path)
    if kept == 0:
        # Still treat export/inference as valid; save the empty prediction image as proof.
        return


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"fod_rtdetrv2_{args.variant}_{args.epochs}e_smoke"
    smoke_output_dir = args.output_dir.resolve() / run_name
    smoke_output_dir.mkdir(parents=True, exist_ok=True)

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
    smoke_config = write_smoke_config(experiment_config, run_name)

    sys.path.insert(0, str(repo_pytorch_dir))
    from src.misc import dist_utils
    from src.core import YAMLConfig
    from src.solver import TASKS
    from src.solver.det_engine import evaluate, train_one_epoch

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=42)

    try:
        cfg = YAMLConfig(
            str(smoke_config),
            device=args.device,
            use_amp=False,
            output_dir=str(smoke_output_dir),
            print_freq=1,
        )
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.train()

        train_batch = next(iter(solver.train_dataloader))
        train_stats = train_one_epoch(
            solver.model,
            solver.criterion,
            [train_batch],
            solver.optimizer,
            solver.device,
            epoch=0,
            max_norm=cfg.clip_max_norm,
            ema=solver.ema,
            scaler=None,
            lr_warmup_scheduler=None,
            writer=None,
            print_freq=1,
        )

        checkpoint_path = smoke_output_dir / "smoke_last.pth"
        torch.save(solver.state_dict(), checkpoint_path)

        val_batch = next(iter(solver.val_dataloader))
        eval_stats, _ = evaluate(
            solver.model,
            solver.criterion,
            solver.postprocessor,
            [val_batch],
            solver.evaluator,
            solver.device,
        )

        onnx_path = export_onnx_model(
            repo_pytorch_dir=repo_pytorch_dir,
            config_path=smoke_config,
            checkpoint_path=checkpoint_path,
            output_path=smoke_output_dir / "smoke_last.onnx",
            input_size=640,
            simplify=False,
        )

        sample_image = first_image(args.dataset_root.resolve() / "val")
        final_inference_output = smoke_output_dir / "smoke_onnxruntime_result.jpg"
        draw_onnx_predictions(
            image_path=sample_image,
            onnx_path=onnx_path,
            output_path=final_inference_output,
        )

        summary = {
            "status": "passed",
            "device": args.device,
            "repo_pytorch_dir": str(repo_pytorch_dir),
            "dataset_root": str(args.dataset_root.resolve()),
            "smoke_config": str(smoke_config.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "onnx": str(onnx_path.resolve()),
            "onnxruntime_result": str(final_inference_output.resolve()),
            "train_loss_keys": sorted(train_stats.keys()),
            "eval_metric_keys": sorted(eval_stats.keys()),
        }
        summary_path = smoke_output_dir / "smoke_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        dist_utils.cleanup()


if __name__ == "__main__":
    main()
