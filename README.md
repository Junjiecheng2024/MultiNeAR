# MultiNeAR: Multi-Class Neural Annotation Refinement for Cardiac CT

This repository contains an implementation of **Neural Annotation Refinement (NeAR)** adapted for multi-class cardiac CT segmentation. NeAR is an implicit neural representation approach that learns to refine noisy or imperfect annotations by modeling anatomical structures as continuous implicit functions.

## Overview

NeAR addresses the challenge of annotation quality in medical image segmentation. Rather than accepting noisy labels as ground truth, NeAR learns a latent representation for each sample that encodes the true underlying anatomy. This approach is particularly effective for:

- **Refining junior annotator labels** to expert-level quality
- **Correcting systematic annotation artifacts**
- **Handling class imbalance** in medical segmentation (e.g., small structures like coronary arteries)
- **Improving downstream segmentation** model training data

### Key Features

- **Sample-specific latent codes**: Each training sample has its own learnable latent vector
- **Appearance-aware refinement**: Integrates CT intensity information for context
- **Multi-resolution training**: Efficient 64³ training with 128³ evaluation
- **Class-balanced loss**: Weighted CE + Dice loss to handle severe class imbalance
- **11-class cardiac structures**: Background, myocardium, LA, LV, RA, RV, aorta, PA, LAA, coronary arteries, PV

## Code Structure

```
MultiNeAR/
├── near/                          # Core NeAR library
│   ├── datasets/                  # Dataset implementations
│   │   ├── refine_dataset.py      # CardiacMultiClassDataset
│   │   ├── prepare_near_data.py   # Data preprocessing script
│   │   ├── split_dataset.py       # Train/val split utility
│   │   └── validate_data.py       # Data validation checks
│   ├── models/                    # Model architectures
│   │   ├── nn3d/                  # 3D neural network modules
│   │   │   ├── model.py           # EmbeddingDecoder (main model)
│   │   │   ├── blocks.py          # Neural building blocks
│   │   │   └── grid.py            # Grid sampling utilities
│   │   └── losses.py              # Combined loss function (CE + Dice + L2)
│   └── utils/                     # Utility functions
│       ├── misc.py                # General utilities
│       ├── multicore.py           # Parallel processing
│       ├── plot3d.py              # 3D visualization
│       └── preprocessing.py       # Data preprocessing utils
└── near_repairing/                # Training and inference scripts
    ├── near_repair.py             # Main training script
    ├── config_cardiac.py          # Configuration file
    ├── inference.py               # Inference and evaluation
    ├── analyze_inference_results.py  # Result analysis
    └── check_class_distribution.py   # Distribution checker
```

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/Junjiecheng2024/MultiNeAR
cd MultiNeAR

# Create conda environment
conda create -n near python=3.9
conda activate near

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install nibabel numpy pandas tqdm matplotlib scikit-image scipy

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### 1. Data Preparation

Prepare your cardiac CT data in the following structure:

```
data_root/
├── appearance/          # CT images (.npy files, 128³)
│   ├── case_001.npy
│   ├── case_002.npy
│   └── ...
├── shape/               # Segmentation labels (.npy files, 128³, int 0-10)
│   ├── case_001.npy
│   ├── case_002.npy
│   └── ...
└── info.csv             # Sample IDs (column: 'id')
```

**Class labels**:
- 0: Background
- 1: Myocardium
- 2: Left Atrium (LA)
- 3: Left Ventricle (LV)
- 4: Right Atrium (RA)
- 5: Right Ventricle (RV)
- 6: Aorta
- 7: Pulmonary Artery (PA)
- 8: Left Atrial Appendage (LAA)
- 9: Coronary Arteries
- 10: Pulmonary Veins (PV)

Run preprocessing (if starting from NIfTI files):

```bash
cd near/datasets
python prepare_near_data.py --input_dir /path/to/nifti --output_dir /path/to/preprocessed
```

### 2. Training

Basic training:

```bash
cd near_repairing
python near_repair.py
```

With custom config:

```bash
export NEAR_CONFIG=config_cardiac
python near_repair.py
```

Background training (recommended):

```bash
nohup python near_repair.py > train.log 2>&1 &
tail -f train.log
```

**Training outputs** (saved to `runs/cardiac_near_YYMMDD_HHMMSS/`):
- `config.json`: Configuration backup
- `latest.pth`: Most recent checkpoint
- `best.pth`: Best checkpoint (lowest validation loss)

### 3. Inference

Run inference on the trained model:

```bash
python inference.py \
    --model_path runs/cardiac_near_251017_143025/best.pth \
    --output_dir inference_results
```

**Inference outputs**:
- `repaired_segmentations/`: Refined segmentation masks (.nii.gz)
- `evaluation_results.csv`: Per-sample Dice scores
- `visualizations/`: Comparison images (optional)

### 4. Result Analysis

Analyze inference results:

```bash
python analyze_inference_results.py \
    --csv inference_results/evaluation_results.csv \
    --output inference_results/analysis
```

**Analysis outputs**:
- Per-class Dice statistics (mean ± std)
- Box plots and histograms
- Markdown report with key findings

## Configuration

Key parameters in `config_cardiac.py`:

```python
cfg = dict(
    # Data
    data_path="/path/to/preprocessed/data",
    num_classes=11,
    
    # Resolution
    target_resolution=128,      # Full resolution (128³)
    training_resolution=64,     # Training resolution (64³ for speed)
    
    # Training
    n_epochs=100,
    batch_size=8,               # Adjust based on GPU memory
    lr=1e-3,
    
    # Loss weights
    lambda_dice=0.2,            # Dice loss weight
    lambda_latent=1e-2,         # L2 regularization on latent codes
    
    # Class weights (handle imbalance)
    class_weights=[1.0, 32.2, 50.2, ...],  # See config file for full list
)
```


## Methodology

### Model Architecture

**EmbeddingDecoder** consists of:
1. **Sample Embedding Table**: Learnable 256-dim latent code per training sample
2. **Appearance Encoder**: Process CT intensity at query locations
3. **Implicit Decoder**: MLP that maps (latent, position, appearance) → class logits

### Training Procedure

1. **Grid Sampling**: Sample 64³ query points from 128³ volume (with noise augmentation)
2. **Forward Pass**: Predict 11-class logits at each query point
3. **Loss Computation**:
   - Cross-Entropy with class weights
   - Multi-class Dice loss (macro-average)
   - L2 regularization on latent codes
4. **Optimization**: Adam optimizer with MultiStepLR scheduler

### Loss Function

```
Total Loss = CE_loss + λ_dice × Dice_loss + λ_latent × ||z||²

where:
- CE_loss: Weighted cross-entropy (handles class imbalance)
- Dice_loss: Multi-class Dice (improves small structure segmentation)
- ||z||²: L2 norm of latent codes (prevents overfitting)
```

## Results

### Example Visualization

NeAR refines noisy annotations by learning implicit representations of cardiac structures:

![Example Result](assets/example_result.png)

*Visualization of annotation refinement: (Left) Original CT slice, (Middle) Original segmentation, (Right) NeAR-refined segmentation. The refined result shows smoother boundaries and better anatomical consistency.*

### Qualitative Improvements

- **Smoother boundaries**: Reduced annotation artifacts
- **Better topology**: More anatomically plausible structures
- **Consistent small structures**: Improved coronary artery and LAA segmentation

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# Reduce batch size or training resolution in config_cardiac.py
batch_size=4  # instead of 8
training_resolution=48  # instead of 64
```

**2. Class imbalance issues**
```python
# Adjust class weights
class_weights=[1.0, 50.0, 100.0, ...]  # Increase weights for rare classes
```

**3. Training divergence**
```python
# Reduce learning rate
lr=5e-4  # instead of 1e-3
# Or increase regularization
lambda_latent=5e-2  # instead of 1e-2
```

## Citation

This implementation is adapted from the original NeAR paper:

```bibtex
@inproceedings{yang2022neural,
  title={Neural Annotation Refinement: Development of a New 3D Dataset for Adrenal Gland Analysis},
  author={Yang, Jiancheng and Shi, Rui and Wickramasinghe, Udaranga and Zhu, Qikui and Ni, Bingbing and Fua, Pascal},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={503--513},
  year={2022},
  organization={Springer}
}
```

## References

- Original NeAR Paper: [arXiv:2206.15328](https://arxiv.org/abs/2206.15328)
- Original NeAR Code: [github.com/HINTLab/NeAR](https://github.com/HINTLab/NeAR)
- Public Cardiac CT Dataset: [github.com/Bjonze/Public-Cardiac-CT-Dataset](https://github.com/Bjonze/Public-Cardiac-CT-Dataset)

## License

Apache License 2.0 (following original NeAR repository)

