# FOD Single-Class RTDETRv2

This repo turns the FOD-A Pascal VOC dataset into a single-class object detector where every object is relabeled as `FOD`, then trains an RTDETRv2 model for 20 epochs.

## What this project does

- Extracts `archive.zip`
- Rewrites every annotation label to `FOD`
- Keeps the official held-out `test.txt` split
- Splits the official `trainval.txt` into reproducible `train` and `val` subsets
- Builds Pascal VOC, YOLO, and COCO exports from the relabeled data
- Trains RTDETRv2 with the official PyTorch implementation
- Exports ONNX and TensorRT deployment artifacts after training

## Important note about TensorFlow vs CUDA RTDETRv2

RTDETRv2 is published in the PyTorch ecosystem, not as a native TensorFlow training stack. Because of that, this project now uses the official RT-DETR repository's `rtdetrv2_pytorch` code for training.

If your real goal is the fastest NVIDIA GPU inference, use the exported TensorRT engine:

- `best.engine` is the preferred CUDA deployment artifact
- `best.onnx` is included as a portable fallback

In practice, TensorRT is the CUDA-optimized path. TensorFlow weights are not the native RTDETRv2 deployment route.

## Recommended environment

Use Linux with an NVIDIA GPU and Python 3.11.

Do not use Python 3.13 for this stack yet. TensorFlow support is still lagging there.

## Install

Install a CUDA-enabled PyTorch build first, then install the rest:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
git clone https://github.com/lyuwenyu/RT-DETR.git third_party/RT-DETR
pip install -r third_party/RT-DETR/rtdetrv2_pytorch/requirements.txt
```

## 1. Prepare the single-class dataset

```bash
python scripts/prepare_fod_dataset.py \
  --archive archive.zip \
  --extract-dir extracted_dataset \
  --output-dir datasets/fod_single_class
```

By default this keeps the official test split, cleans duplicate/overlapping split IDs, fills in any annotations missing from the shipped split files, and then splits the remaining `trainval` pool with a 90/10 train/val split using seed `42`.

With the current dataset layout this produces:

- `train`: 22,828 images
- `val`: 2,536 images
- `test`: 8,429 images

Generated files:

- `datasets/fod_single_class/fod.yaml`
- `datasets/fod_single_class/dataset_summary.json`
- `datasets/fod_single_class/coco/...`
- `datasets/fod_single_class/voc/...`
- `datasets/fod_single_class/yolo/...`

## 2. Train RTDETRv2 for 20 epochs

```bash
python scripts/train_rtdetr.py \
  --repo-dir third_party/RT-DETR \
  --dataset-root datasets/fod_single_class/coco \
  --variant l \
  --epochs 20 \
  --output-dir runs/fod_rtdetrv2
```

This script:

- links the prepared COCO dataset into the official RTDETRv2 repo
- writes generated dataset and experiment configs into that repo
- trains for 20 epochs with AMP enabled by default
- exports ONNX and TensorRT artifacts after training

If you want a lighter model, switch `--variant` from `l` to `s`.

## 3. Run inference

Use the fastest artifact you have available.

TensorRT engine:

```bash
python scripts/predict_fod.py \
  --repo-dir third_party/RT-DETR \
  --backend tensorrt \
  --model runs/fod_rtdetrv2/fod_rtdetrv2_l_20e/exports/best_stg2.engine \
  --source path/to/image.jpg \
  --output runs/fod_predict/result_trt.jpg
```

PyTorch checkpoint:

```bash
python scripts/predict_fod.py \
  --repo-dir third_party/RT-DETR \
  --backend torch \
  --config third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/fod_rtdetrv2_l_20e.yml \
  --checkpoint runs/fod_rtdetrv2/fod_rtdetrv2_l_20e/best_stg2.pth \
  --source path/to/image.jpg \
  --output runs/fod_predict/result_torch.jpg
```

ONNX Runtime fallback:

```bash
python scripts/predict_fod.py \
  --repo-dir third_party/RT-DETR \
  --backend onnxruntime \
  --model runs/fod_rtdetrv2/fod_rtdetrv2_l_20e/exports/best_stg2.onnx \
  --source path/to/image.jpg \
  --output runs/fod_predict/result_onnx.jpg
```

## Output structure

```text
datasets/fod_single_class/
  classes.txt
  coco/
  dataset_summary.json
  fod.yaml
  voc/
  yolo/

runs/fod_rtdetrv2/
  fod_rtdetrv2_l_20e/
    best_stg2.pth
    exports/
      best_stg2.engine
      best_stg2.onnx
```

## Notes

- The dataset prep script relabels every original class name to the single label `FOD`.
- The training path is now aligned to official RTDETRv2 rather than the earlier Ultralytics RT-DETR wrapper.
- Checkpoint filenames can differ slightly by official repo version, so the export wrapper auto-discovers the newest `.pth` if needed.
- For deployment, use TensorRT or ONNX artifacts instead of the raw training checkpoint whenever possible.
