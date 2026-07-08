#!/usr/bin/env python3
"""Shared helpers for the RTDETRv2 training/export wrappers."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


DEFAULT_DATASET_NAME = "fod_single_class"
DEFAULT_REPO_DIR = Path("third_party/RT-DETR")


def path_lexists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def resolve_repo_pytorch_dir(repo_dir: Path) -> Path:
    candidate = repo_dir.resolve()

    if (candidate / "tools" / "train.py").is_file() and (candidate / "configs").is_dir():
        return candidate

    nested = candidate / "rtdetrv2_pytorch"
    if (nested / "tools" / "train.py").is_file() and (nested / "configs").is_dir():
        return nested

    raise SystemExit(
        "RTDETRv2 repo not found.\n"
        f"Expected either:\n"
        f"  - {(candidate / 'tools/train.py')}\n"
        f"  - {(nested / 'tools/train.py')}\n\n"
        "Clone the official repo first, for example:\n"
        f"  git clone https://github.com/lyuwenyu/RT-DETR.git {repo_dir}"
    )


def ensure_coco_dataset_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    required_paths = [
        dataset_root / "annotations" / "instances_train.json",
        dataset_root / "annotations" / "instances_val.json",
        dataset_root / "annotations" / "instances_test.json",
        dataset_root / "train",
        dataset_root / "val",
        dataset_root / "test",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(
            f"Prepared COCO dataset not found at {dataset_root}.\n"
            f"Missing: {missing[0]}\n"
            "Run scripts/prepare_fod_dataset.py first."
        )
    return dataset_root


def ensure_repo_dataset_link(repo_pytorch_dir: Path, dataset_root: Path, dataset_name: str) -> Path:
    dataset_dir = repo_pytorch_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    link_path = dataset_dir / dataset_name
    target = dataset_root.resolve()

    if path_lexists(link_path):
        try:
            current = link_path.resolve()
        except FileNotFoundError:
            current = None
        if current == target:
            return link_path
        raise SystemExit(
            f"Dataset link already exists and points elsewhere: {link_path}\n"
            f"Current target: {current}\n"
            f"Expected target: {target}\n"
            "Please remove or fix that link/path, then rerun."
        )

    os.symlink(target, link_path, target_is_directory=True)
    return link_path


def variant_config_overrides(variant: str, epochs: int) -> dict:
    stop_epoch = max(1, epochs - 3)
    overrides: dict = {
        "epoches": epochs,
        "train_dataloader": {
            "dataset": {
                "transforms": {
                    "policy": {
                        "epoch": stop_epoch,
                    }
                }
            },
            "collate_fn": {
                "stop_epoch": stop_epoch,
            },
        },
    }

    if variant == "s":
        overrides.update(
            {
                "PResNet": {
                    "depth": 18,
                    "freeze_at": -1,
                    "freeze_norm": False,
                    "pretrained": True,
                },
                "HybridEncoder": {
                    "in_channels": [128, 256, 512],
                    "hidden_dim": 256,
                    "expansion": 0.5,
                },
                "RTDETRTransformerv2": {
                    "num_layers": 3,
                },
            }
        )

    return overrides


def write_dataset_config(repo_pytorch_dir: Path, dataset_name: str, workers: int) -> Path:
    config_path = repo_pytorch_dir / "configs" / "dataset" / f"{dataset_name}_detection.yml"
    config = {
        "task": "detection",
        "evaluator": {
            "type": "CocoEvaluator",
            "iou_types": ["bbox"],
        },
        "num_classes": 1,
        "remap_mscoco_category": False,
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": f"./dataset/{dataset_name}/train/",
                "ann_file": f"./dataset/{dataset_name}/annotations/instances_train.json",
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": None,
                },
            },
            "shuffle": True,
            "num_workers": workers,
            "drop_last": True,
            "collate_fn": {
                "type": "BatchImageCollateFunction",
            },
        },
        "val_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": f"./dataset/{dataset_name}/val/",
                "ann_file": f"./dataset/{dataset_name}/annotations/instances_val.json",
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": None,
                },
            },
            "shuffle": False,
            "num_workers": max(1, workers // 2),
            "drop_last": False,
            "collate_fn": {
                "type": "BatchImageCollateFunction",
            },
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def write_experiment_config(
    repo_pytorch_dir: Path,
    dataset_name: str,
    variant: str,
    epochs: int,
    run_name: str,
) -> Path:
    if variant not in {"l", "s"}:
        raise SystemExit("Only RTDETRv2 variants 'l' and 's' are currently scaffolded.")

    config_path = repo_pytorch_dir / "configs" / "rtdetrv2" / f"{run_name}.yml"
    config = {
        "__include__": [
            f"../dataset/{dataset_name}_detection.yml",
            "../runtime.yml",
            "./include/dataloader.yml",
            "./include/optimizer.yml",
            "./include/rtdetrv2_r50vd.yml",
        ],
        "output_dir": f"./output/{run_name}",
    }
    config.update(variant_config_overrides(variant=variant, epochs=epochs))

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def prepare_rtdetrv2_workspace(
    repo_dir: Path,
    dataset_root: Path,
    dataset_name: str,
    variant: str,
    epochs: int,
    run_name: str,
    workers: int,
) -> dict[str, Path]:
    repo_pytorch_dir = resolve_repo_pytorch_dir(repo_dir)
    dataset_root = ensure_coco_dataset_root(dataset_root)
    dataset_link = ensure_repo_dataset_link(repo_pytorch_dir, dataset_root, dataset_name)
    dataset_config = write_dataset_config(
        repo_pytorch_dir=repo_pytorch_dir,
        dataset_name=dataset_name,
        workers=workers,
    )
    experiment_config = write_experiment_config(
        repo_pytorch_dir=repo_pytorch_dir,
        dataset_name=dataset_name,
        variant=variant,
        epochs=epochs,
        run_name=run_name,
    )
    return {
        "repo_pytorch_dir": repo_pytorch_dir,
        "dataset_root": dataset_root,
        "dataset_link": dataset_link,
        "dataset_config": dataset_config,
        "experiment_config": experiment_config,
    }


def find_checkpoint(output_dir: Path) -> Path:
    preferred_names = [
        "best_stg2.pth",
        "best_stg1.pth",
        "best.pth",
        "last.pth",
        "checkpoint.pth",
    ]
    for name in preferred_names:
        candidate = output_dir / name
        if candidate.exists():
            return candidate

    pth_files = sorted(output_dir.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if pth_files:
        return pth_files[0]

    raise SystemExit(f"No .pth checkpoint was found under {output_dir}")
