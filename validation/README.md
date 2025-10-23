# Validation

This directory contains validation experiments to quantitatively demonstrate the effectiveness of NeAR topology repair on cardiac CT segmentations.

## Overview

We provide two complementary validation approaches:

1. **3D U-Net Validation**: Train downstream segmentation models to show that repaired labels lead to better model performance
2. **Connected Components Analysis**: Quantify topological improvements by measuring fragmentation reduction

## Directory Structure

```
validation/
├── unet3d/                      # 3D U-Net validation
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── resize_labels.py        # Label resizing utility
│   └── README.md
└── connected_components/        # Topology analysis
    ├── analyze.py              # Connected components analysis
    └── README.md
```

## Quick Start

### 3D U-Net Validation

```bash
cd unet3d

# 1. Resize repaired labels to original dimensions
python resize_labels.py

# 2. Train with repaired labels
python train.py --label_type repaired --epochs 50

# 3. Train with baseline labels for comparison
python train.py --label_type baseline --epochs 50

# 4. Evaluate both models
python evaluate.py \
    --checkpoint runs_repaired_improved/best_model.pth \
    --label_type repaired \
    --split_file runs_repaired_improved/train_val_split.json
```

### Connected Components Analysis

```bash
cd connected_components

# Analyze topological changes
python analyze.py

# Results saved to data/cc_stats.csv
```

## Validation Metrics

### 3D U-Net Performance
- **Metric**: Dice Score (per-class and macro-averaged)
- **Hypothesis**: Models trained with NeAR-repaired labels achieve higher Dice scores
- **Evaluation**: Compare validation performance between baseline and repaired training

### Topological Quality
- **Metrics**: 
  - Number of connected components (lower is better)
  - Largest component ratio (higher is better)
- **Hypothesis**: NeAR repair reduces fragmentation and improves structural coherence
- **Evaluation**: Compare statistics before and after repair

## Expected Results

**3D U-Net Validation:**
- Repaired labels → Higher Dice scores on validation set
- Demonstrates that topology repair improves label quality for downstream tasks

**Connected Components Analysis:**
- Repaired segmentations → Fewer components, higher largest ratio
- Quantifies topological improvements directly

## Requirements

See main repository `requirements.txt`. Key dependencies:
- PyTorch (for 3D U-Net)
- nibabel (for NIfTI I/O)
- cc3d (for connected components)
- numpy, scipy

## Citation

If you use these validation methods, please cite the NeAR paper:
```bibtex
@article{near2024,
  title={NeAR: Neural Anatomical Repair for Cardiac CT Segmentation},
  author={...},
  journal={...},
  year={2024}
}
```
