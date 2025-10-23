# 3D U-Net Validation

This directory contains scripts for validating the NeAR-repaired cardiac CT segmentations using a downstream 3D U-Net model.

## Overview

We train a 3D U-Net on both baseline (original) and NeAR-repaired segmentations to evaluate whether the repaired labels lead to better model performance. This provides quantitative evidence of segmentation quality improvement.

## Files

- `train.py`: Train 3D U-Net with baseline or repaired labels
- `evaluate.py`: Evaluate trained model on validation set
- `resize_labels.py`: Resize 128³ repaired labels to original image dimensions

## Training

Train with repaired labels:
```bash
python train.py \
    --label_type repaired \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --val_ratio 0.2 \
    --fg_ratio 0.7
```

Train with baseline labels:
```bash
python train.py \
    --label_type baseline \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3
```

## Evaluation

Evaluate a trained model:
```bash
python evaluate.py \
    --checkpoint runs_repaired_improved/best_model.pth \
    --label_type repaired \
    --split_file runs_repaired_improved/train_val_split.json \
    --output results_repaired.json
```

## Resizing Labels

Resize NeAR-repaired labels (128³) to match original image dimensions:
```bash
python resize_labels.py
```

This step is required before training with repaired labels.

## Model Architecture

- **Architecture**: 3D U-Net with InstanceNorm
- **Input**: Single-channel CT volumes (HU clipping: -1000 to 1000)
- **Output**: 11-class segmentation
- **Loss**: CrossEntropy + Dice Loss
- **Features**:
  - Foreground-aware patch sampling
  - Sliding window inference
  - Per-class Dice score evaluation

## Expected Results

Models trained with NeAR-repaired labels should achieve higher Dice scores on validation sets compared to models trained with baseline labels, demonstrating the effectiveness of the topology repair.
