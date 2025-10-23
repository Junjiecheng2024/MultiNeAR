#!/usr/bin/env python3
"""
NeAR Inference and Evaluation Script

Performs inference on trained NeAR model, generates refined segmentations,
and evaluates performance using Dice scores and other metrics.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from near.models.nn3d.model import EmbeddingDecoder
from near.models.nn3d.grid import GatherGridsFromVolumes
from near.datasets.refine_dataset import CardiacMultiClassDataset

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed. Can only save .npy format.")


def compute_dice_per_class(pred, gt, num_classes=11, smooth=1e-6):
    """Compute Dice score for each class.
    
    Args:
        pred: Predicted segmentation [D,H,W], integer labels 0~K-1
        gt: Ground truth segmentation [D,H,W], integer labels 0~K-1
        num_classes: Number of classes (including background)
        smooth: Smoothing factor to prevent division by zero
    
    Returns:
        dice_scores: Array of Dice scores per class [K]
    """
    dice_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        
        intersection = np.sum(pred_c * gt_c)
        denominator = np.sum(pred_c) + np.sum(gt_c)
        
        if denominator == 0:
            dice = np.nan
        else:
            dice = (2.0 * intersection + smooth) / (denominator + smooth)
        
        dice_scores.append(dice)
    
    return np.array(dice_scores)


def to_device(x, device):
    """Move data to device recursively."""
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(xx, device) for xx in x)
    return x.to(device) if hasattr(x, "to") else x


@torch.no_grad()
def sample_labels_nearest(seg_long, grids):
    """Sample labels at grid points using nearest neighbor interpolation.
    
    Args:
        seg_long: Segmentation labels [B,1,D,H,W]
        grids: Sampling grid coordinates [B,Ds,Hs,Ws,3]
    
    Returns:
        labels: Sampled labels [B,Ds,Hs,Ws]
    """
    lab = F.grid_sample(
        seg_long.float(),
        grids,
        mode='nearest',
        align_corners=True
    )
    return lab.squeeze(1).long()


def run_inference(model, dataset, gather_fn, device, output_dir, 
                  save_format='nii.gz', save_visualization=False, vis_num=10):
    """Run inference on all samples and save results.
    
    Args:
        model: Trained NeAR model
        dataset: Dataset to run inference on
        gather_fn: Grid sampling function
        device: Computation device
        output_dir: Output directory path
        save_format: Output format ('npy' or 'nii.gz')
        save_visualization: Whether to save visualization
        vis_num: Number of samples to visualize
    
    Returns:
        results_df: DataFrame with evaluation results
    """
    model.eval()
    
    output_path = Path(output_dir)
    seg_dir = output_path / "repaired_segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    if save_visualization:
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print("\n" + "=" * 80)
    print(f"Running inference on {len(dataset)} samples...")
    print("=" * 80)
    
    for idx in tqdm(range(len(dataset)), desc="Inference"):
        # Load data
        idx_tensor, app, seg = dataset[idx]
        
        # Move to device
        idx_tensor = to_device(torch.tensor([idx_tensor]), device)
        app = to_device(app.unsqueeze(0), device)
        seg = to_device(seg.unsqueeze(0), device)
        
        # Generate grid and sample
        _, grids, app_label = gather_fn(app)
        
        # Model inference
        logits, _ = model(idx_tensor, grids, app_label)
        
        # Get prediction
        pred = torch.argmax(logits, dim=1)  # [B,D,H,W]
        
        # Move to CPU and convert to numpy
        pred_np = pred.cpu().numpy()[0]  # [D,H,W]
        seg_np = seg.cpu().numpy()[0, 0]  # [D,H,W]
        
        # Compute metrics
        dice_scores = compute_dice_per_class(pred_np, seg_np, num_classes=11)
        
        # Prepare result entry
        result = {'case_id': f"case_{idx:03d}"}
        for c in range(11):
            result[f'dice_class_{c}'] = dice_scores[c]
        result['dice_mean'] = np.nanmean(dice_scores[1:])  # Exclude background
        
        results.append(result)
        
        # Save segmentation
        if save_format == 'npy':
            save_path = seg_dir / f"case_{idx:03d}.npy"
            np.save(save_path, pred_np)
        elif save_format == 'nii.gz' and HAS_NIBABEL:
            save_path = seg_dir / f"case_{idx:03d}.nii.gz"
            nib_img = nib.Nifti1Image(pred_np.astype(np.int16), affine=np.eye(4))
            nib.save(nib_img, save_path)
        
        # Save visualization
        if save_visualization and idx < vis_num:
            save_comparison_slice(seg_np, pred_np, vis_dir, idx)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = output_path / "evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    class_names = ["Background", "Myocardium", "LA", "LV", "RA", "RV", 
                   "Aorta", "PA", "LAA", "Coronary", "PV"]
    
    for c in range(11):
        col = f'dice_class_{c}'
        if col in results_df.columns:
            mean_dice = results_df[col].mean()
            std_dice = results_df[col].std()
            print(f"Class {c:2d} ({class_names[c]:12s}): "
                  f"Dice = {mean_dice:.4f} ± {std_dice:.4f}")
    
    overall_mean = results_df['dice_mean'].mean()
    overall_std = results_df['dice_mean'].std()
    print(f"\nOverall Mean Dice (Classes 1-10): {overall_mean:.4f} ± {overall_std:.4f}")
    
    return results_df


def save_comparison_slice(original, refined, output_dir, case_idx, slice_idx=None):
    """Save visualization comparing original and refined segmentations.
    
    Args:
        original: Original segmentation [D,H,W]
        refined: Refined segmentation [D,H,W]
        output_dir: Output directory
        case_idx: Case index
        slice_idx: Slice index (middle slice if None)
    """
    if slice_idx is None:
        slice_idx = original.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original[slice_idx], cmap='tab20', vmin=0, vmax=10)
    axes[0].set_title('Original Segmentation')
    axes[0].axis('off')
    
    # Refined
    axes[1].imshow(refined[slice_idx], cmap='tab20', vmin=0, vmax=10)
    axes[1].set_title('Refined Segmentation')
    axes[1].axis('off')
    
    # Difference
    diff = (original[slice_idx] != refined[slice_idx]).astype(np.float32)
    axes[2].imshow(diff, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Differences')
    axes[2].axis('off')
    
    plt.suptitle(f'Case {case_idx:03d} - Slice {slice_idx}')
    plt.tight_layout()
    
    save_path = output_dir / f"case_{case_idx:03d}_slice{slice_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='NeAR Inference Script')
    parser.add_argument('--config', type=str, default='config_cardiac',
                       help='Config module name (without .py)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory')
    parser.add_argument('--save_format', type=str, default='nii.gz',
                       choices=['npy', 'nii.gz'],
                       help='Output segmentation format')
    parser.add_argument('--save_visualization', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--vis_num', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    import importlib
    cfg_module = importlib.import_module(args.config)
    cfg = cfg_module.cfg
    
    print("=" * 80)
    print("NeAR Inference Script")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.save_format}")
    print("=" * 80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = CardiacMultiClassDataset(
        root=cfg["data_path"],
        resolution=cfg["target_resolution"],
        n_samples=cfg.get("n_training_samples", None),
        normalize=True
    )
    print(f"Loaded dataset: {len(dataset)} samples")
    
    # Initialize grid sampler
    gather_fn = GatherGridsFromVolumes(
        cfg["target_resolution"],
        grid_noise=None
    )
    
    # Initialize model
    model = EmbeddingDecoder(
        latent_dimension=cfg.get("latent_dim", 256),
        n_samples=len(dataset),
        num_classes=cfg["num_classes"],
        appearance=cfg.get("appearance", True)
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {args.model_path}")
    
    # Run inference
    results_df = run_inference(
        model, dataset, gather_fn, device,
        args.output_dir, args.save_format,
        args.save_visualization, args.vis_num
    )
    
    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print("Generated files:")
    print("  - evaluation_results.csv")
    print(f"  - repaired_segmentations/ ({len(dataset)} files)")
    if args.save_visualization:
        print(f"  - visualizations/ ({args.vis_num} files)")


if __name__ == "__main__":
    main()
