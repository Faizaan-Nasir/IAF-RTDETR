#!/usr/bin/env python3
"""Prepare the FOD-A dataset as a single-class RT-DETR training set.

This script:
1. Extracts `archive.zip` if needed.
2. Uses the official Pascal VOC `trainval.txt` and `test.txt` files.
3. Splits `trainval` into reproducible `train` and `val` subsets.
4. Rewrites every object label to the single class `FOD`.
5. Exports both:
   - relabeled Pascal VOC annotations
   - YOLO-format labels
   - COCO-format annotations for official RTDETRv2 training
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import yaml


FOD_CLASS_NAME = "FOD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("archive.zip"),
        help="Path to the dataset archive.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("extracted_dataset"),
        help="Directory where the archive should be extracted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/fod_single_class"),
        help="Directory where the relabeled dataset should be written.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction of the official trainval split to reserve for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when splitting train/val.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Delete and re-extract the archive.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete and rebuild the prepared dataset output directory.",
    )
    return parser.parse_args()


def extract_archive(archive_path: Path, extract_dir: Path, force_extract: bool) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if force_extract and extract_dir.exists():
        shutil.rmtree(extract_dir)

    if extract_dir.exists() and any(extract_dir.iterdir()):
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)


def find_voc_root(extract_dir: Path) -> Path:
    for candidate in extract_dir.rglob("VOC2007"):
        if (
            (candidate / "Annotations").is_dir()
            and (candidate / "JPEGImages").is_dir()
            and (candidate / "ImageSets" / "Main" / "trainval.txt").is_file()
            and (candidate / "ImageSets" / "Main" / "test.txt").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate a Pascal VOC dataset root under "
        f"{extract_dir.resolve()}"
    )


def read_unique_split_ids(path: Path) -> tuple[list[str], int]:
    seen: set[str] = set()
    ids: list[str] = []
    duplicate_count = 0

    for raw_line in path.read_text().splitlines():
        image_id = raw_line.strip()
        if not image_id:
            continue
        if image_id in seen:
            duplicate_count += 1
            continue
        seen.add(image_id)
        ids.append(image_id)

    if not ids:
        raise ValueError(f"Split file is empty: {path}")
    return ids, duplicate_count


def split_train_val(
    trainval_ids: list[str], val_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")

    shuffled = list(trainval_ids)
    random.Random(seed).shuffle(shuffled)

    raw_val_count = int(round(len(shuffled) * val_ratio))
    val_count = max(1, min(len(shuffled) - 1, raw_val_count))

    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    return train_ids, val_ids


def ensure_clean_dir(path: Path, force_rebuild: bool) -> None:
    if force_rebuild and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            os.symlink(src.resolve(), dst)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def yolo_line_from_bbox(
    class_id: int, width: int, height: int, xmin: float, ymin: float, xmax: float, ymax: float
) -> str:
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{box_width:.6f} "
        f"{box_height:.6f}"
    )


def parse_and_relabel_annotation(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    width = int(float(root.findtext("size/width", default="0")))
    height = int(float(root.findtext("size/height", default="0")))
    if not filename or width <= 0 or height <= 0:
        raise ValueError(f"Invalid annotation metadata in {xml_path}")

    objects = []
    original_names: list[str] = []
    skipped_boxes = 0

    for obj in root.findall("object"):
        original_name = (obj.findtext("name") or "UNKNOWN").strip()
        original_names.append(original_name)

        name_node = obj.find("name")
        if name_node is not None:
            name_node.text = FOD_CLASS_NAME

        bbox = obj.find("bndbox")
        if bbox is None:
            skipped_boxes += 1
            continue

        xmin = clamp(float(bbox.findtext("xmin", default="0")), 0.0, float(width))
        ymin = clamp(float(bbox.findtext("ymin", default="0")), 0.0, float(height))
        xmax = clamp(float(bbox.findtext("xmax", default="0")), 0.0, float(width))
        ymax = clamp(float(bbox.findtext("ymax", default="0")), 0.0, float(height))

        if xmax <= xmin or ymax <= ymin:
            skipped_boxes += 1
            continue

        objects.append(
            {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return {
        "tree": tree,
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
        "original_names": original_names,
        "skipped_boxes": skipped_boxes,
    }


def write_split_file(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(ids)
    if contents:
        contents += "\n"
    path.write_text(contents)


def build_coco_annotations(
    split_ids: Iterable[str],
    filename_cache: dict[str, str],
    annotation_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id in split_ids:
        annotation = annotation_cache[image_id]
        filename = filename_cache[image_id]
        numeric_image_id = int(image_id)

        images.append(
            {
                "id": numeric_image_id,
                "file_name": filename,
                "width": annotation["width"],
                "height": annotation["height"],
            }
        )

        for obj in annotation["objects"]:
            box_width = obj["xmax"] - obj["xmin"]
            box_height = obj["ymax"] - obj["ymin"]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": numeric_image_id,
                    "category_id": 1,
                    "bbox": [obj["xmin"], obj["ymin"], box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": FOD_CLASS_NAME,
                "supercategory": FOD_CLASS_NAME,
            }
        ],
    }


def main() -> None:
    args = parse_args()

    extract_archive(args.archive, args.extract_dir, args.force_extract)
    voc_root = find_voc_root(args.extract_dir)

    all_annotation_ids = sorted(path.stem for path in (voc_root / "Annotations").glob("*.xml"))
    all_annotation_id_set = set(all_annotation_ids)

    trainval_ids, trainval_duplicate_entries = read_unique_split_ids(
        voc_root / "ImageSets" / "Main" / "trainval.txt"
    )
    test_ids, test_duplicate_entries = read_unique_split_ids(
        voc_root / "ImageSets" / "Main" / "test.txt"
    )

    trainval_unknown_ids = [image_id for image_id in trainval_ids if image_id not in all_annotation_id_set]
    test_unknown_ids = [image_id for image_id in test_ids if image_id not in all_annotation_id_set]
    trainval_ids = [image_id for image_id in trainval_ids if image_id in all_annotation_id_set]
    test_ids = [image_id for image_id in test_ids if image_id in all_annotation_id_set]

    overlap_ids = sorted(set(trainval_ids) & set(test_ids))
    overlap_id_set = set(overlap_ids)
    if overlap_ids:
        trainval_ids = [image_id for image_id in trainval_ids if image_id not in overlap_id_set]

    unassigned_ids = sorted(all_annotation_id_set - set(trainval_ids) - set(test_ids))
    trainval_pool = trainval_ids + unassigned_ids

    train_ids, val_ids = split_train_val(trainval_pool, args.val_ratio, args.seed)

    split_to_ids = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    overlap = (
        set(train_ids) & set(val_ids)
        or set(train_ids) & set(test_ids)
        or set(val_ids) & set(test_ids)
    )
    if overlap:
        raise ValueError(f"Found overlapping image IDs between splits: {sorted(overlap)[:5]}")

    output_dir = args.output_dir.resolve()
    ensure_clean_dir(output_dir, args.force_rebuild)

    voc_output = output_dir / "voc"
    yolo_output = output_dir / "yolo"
    coco_output = output_dir / "coco"
    voc_annotations_dir = voc_output / "Annotations"
    voc_images_dir = voc_output / "JPEGImages"
    voc_imagesets_dir = voc_output / "ImageSets" / "Main"
    coco_annotations_dir = coco_output / "annotations"

    original_class_counts: Counter[str] = Counter()
    split_box_counts: dict[str, int] = defaultdict(int)
    skipped_boxes = 0
    link_modes: Counter[str] = Counter()
    missing_images: list[str] = []

    all_ids = all_annotation_ids
    annotation_cache: dict[str, dict[str, Any]] = {}
    label_cache: dict[str, list[str]] = {}
    filename_cache: dict[str, str] = {}

    for image_id in all_ids:
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        annotation = parse_and_relabel_annotation(xml_path)
        annotation_cache[image_id] = annotation
        filename = annotation["filename"]
        image_path = voc_root / "JPEGImages" / filename

        if not image_path.exists():
            missing_images.append(str(image_path))
            continue

        filename_cache[image_id] = filename
        original_class_counts.update(annotation["original_names"])
        skipped_boxes += int(annotation["skipped_boxes"])

        yolo_lines = [
            yolo_line_from_bbox(
                0,
                annotation["width"],
                annotation["height"],
                obj["xmin"],
                obj["ymin"],
                obj["xmax"],
                obj["ymax"],
            )
            for obj in annotation["objects"]
        ]
        label_cache[image_id] = yolo_lines

        relabeled_xml_path = voc_annotations_dir / f"{image_id}.xml"
        relabeled_xml_path.parent.mkdir(parents=True, exist_ok=True)
        annotation["tree"].write(relabeled_xml_path, encoding="utf-8", xml_declaration=False)
        link_modes.update([link_or_copy(image_path, voc_images_dir / filename)])

    if missing_images:
        raise FileNotFoundError(
            "Some images referenced by annotations are missing. "
            f"First missing file: {missing_images[0]}"
        )

    for split_name, split_ids in split_to_ids.items():
        image_dir = yolo_output / "images" / split_name
        label_dir = yolo_output / "labels" / split_name
        coco_image_dir = coco_output / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        coco_image_dir.mkdir(parents=True, exist_ok=True)

        for image_id in split_ids:
            filename = filename_cache[image_id]
            image_src = voc_root / "JPEGImages" / filename
            image_dst = image_dir / filename
            coco_image_dst = coco_image_dir / filename
            label_dst = label_dir / f"{image_id}.txt"

            link_modes.update([link_or_copy(image_src, image_dst)])
            link_modes.update([link_or_copy(image_src, coco_image_dst)])
            label_lines = label_cache[image_id]
            label_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
            split_box_counts[split_name] += len(label_lines)

        coco_annotations = build_coco_annotations(
            split_ids=split_ids,
            filename_cache=filename_cache,
            annotation_cache=annotation_cache,
        )
        coco_annotations_dir.mkdir(parents=True, exist_ok=True)
        with (coco_annotations_dir / f"instances_{split_name}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(coco_annotations, handle, indent=2)

    write_split_file(voc_imagesets_dir / "train.txt", train_ids)
    write_split_file(voc_imagesets_dir / "val.txt", val_ids)
    write_split_file(voc_imagesets_dir / "test.txt", test_ids)
    write_split_file(voc_imagesets_dir / "trainval.txt", sorted(train_ids + val_ids))

    (output_dir / "classes.txt").write_text(f"{FOD_CLASS_NAME}\n")

    dataset_yaml = {
        "path": str(yolo_output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": [FOD_CLASS_NAME],
        "nc": 1,
    }
    with (output_dir / "fod.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False)

    summary = {
        "source_voc_root": str(voc_root.resolve()),
        "output_dir": str(output_dir),
        "single_class_name": FOD_CLASS_NAME,
        "seed": args.seed,
        "val_ratio_from_trainval": args.val_ratio,
        "total_images": len(all_ids),
        "total_original_boxes": int(sum(original_class_counts.values())),
        "total_relabeled_boxes": int(sum(split_box_counts.values())),
        "skipped_invalid_boxes": skipped_boxes,
        "split_cleanup": {
            "trainval_duplicate_entries_removed": trainval_duplicate_entries,
            "test_duplicate_entries_removed": test_duplicate_entries,
            "trainval_ids_not_found_in_annotations": len(trainval_unknown_ids),
            "test_ids_not_found_in_annotations": len(test_unknown_ids),
            "trainval_test_overlap_kept_in_test_only": len(overlap_ids),
            "annotations_missing_from_split_files_added_to_trainval_pool": len(unassigned_ids),
        },
        "split_counts": {
            "train": {"images": len(train_ids), "boxes": split_box_counts["train"]},
            "val": {"images": len(val_ids), "boxes": split_box_counts["val"]},
            "test": {"images": len(test_ids), "boxes": split_box_counts["test"]},
        },
        "exported_formats": ["voc", "yolo", "coco"],
        "original_class_counts": dict(sorted(original_class_counts.items())),
        "image_link_modes": dict(link_modes),
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSingle-class dataset written to: {output_dir}")
    print(f"Dataset YAML: {output_dir / 'fod.yaml'}")


if __name__ == "__main__":
    main()
