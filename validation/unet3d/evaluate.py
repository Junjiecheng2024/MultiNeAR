#!/usr/bin/env python3
"""Evaluate model checkpoint on validation set."""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
import json
from tqdm import tqdm

sys.path.insert(0, '/home/user/persistent/3dUNet_val')
from train_unet_improved import UNet3D, sliding_window_inference

def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"GPU cache cleared. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = UNet3D(n_channels=1, n_classes=11, base_channels=32)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, image_dir, label_dir, case_ids, device='cuda', 
                   patch_size=(96,96,96), stride=(48,48,48), batch_size=16):
    """Evaluate model on specified cases.
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        label_dir: Directory containing labels
        case_ids: List of case IDs to evaluate
        device: Device for inference
        patch_size: Size of sliding window patches
        stride: Stride for sliding window
        batch_size: Batch size for inference
    
    Returns:
        macro_dice: Average Dice score across all classes
        mean_per_class: Per-class Dice scores
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    dice_scores = []
    per_class_dice_all = []
    
    print(f"\nEvaluating on {len(case_ids)} cases with batch_size={batch_size}...")
    
    for i, case_id in enumerate(tqdm(case_ids, desc="Evaluating")):
        image_path = None
        label_path = None
        
        for name in (f"{case_id}.nii.gz", f"{case_id}.nii.img.nii.gz"):
            p = image_dir / name
            if p.exists():
                image_path = p
                break
        
        for name in (f"{case_id}.nii.gz", f"{case_id}.nii.img.nii.gz"):
            p = label_dir / name
            if p.exists():
                label_path = p
                break
        
        if not image_path or not label_path:
            print(f"Warning: Case {case_id} not found, skipping...")
            continue
        
        image = nib.load(str(image_path)).get_fdata().astype(np.float32)
        label = nib.load(str(label_path)).get_fdata().astype(np.int64)
        
        image = np.clip(image, -1000, 1000)
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        pred = sliding_window_inference(model, image, patch_size=patch_size, 
                                        stride=stride, device=device, n_classes=11,
                                        batch_size=batch_size)
        
        dice_per_class = []
        for c in range(11):
            pred_c = (pred == c)
            label_c = (label == c)
            
            intersection = np.sum(pred_c & label_c)
            union = np.sum(pred_c) + np.sum(label_c)
            
            if union > 0:
                dice = 2.0 * intersection / union
                dice_per_class.append(dice)
        
        if dice_per_class:
            dice_scores.append(np.mean(dice_per_class))
            per_class_dice_all.append(dice_per_class)
    
    macro_dice = np.mean(dice_scores)
    
    mean_per_class = []
    for c in range(11):
        class_dices = []
        for sample_dice in per_class_dice_all:
            if len(sample_dice) > c:
                class_dices.append(sample_dice[c])
        if class_dices:
            mean_per_class.append(np.mean(class_dices))
        else:
            mean_per_class.append(0.0)
    
    return macro_dice, mean_per_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--label_type', type=str, required=True, choices=['baseline', 'repaired'])
    parser.add_argument('--split_file', type=str, required=True, help='Path to train_val_split.json')
    parser.add_argument('--num_cases', type=int, default=None, help='Number of cases to evaluate (None=all)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print(f"Model loaded successfully")
    
    with open(args.split_file, 'r') as f:
        split_info = json.load(f)
    
    val_ids = split_info['val_ids']
    
    if args.num_cases is None:
        args.num_cases = len(val_ids) // 2
        print(f"Using half of validation set: {args.num_cases} cases (out of {len(val_ids)})")
    
    if args.num_cases:
        val_ids = val_ids[:args.num_cases]
    
    print(f"Validation cases: {len(val_ids)}")
    
    image_dir = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/matched_dataset/images"
    
    if args.label_type == 'baseline':
        label_dir = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/matched_dataset/segmentations"
    else:
        label_dir = "/home/user/persistent/3dUNet_val/labelsTr_repaired_resized"
    
    macro_dice, per_class_dice = evaluate_model(
        model, image_dir, label_dir, val_ids, device,
        patch_size=(96,96,96), stride=(72,72,72), batch_size=32
    )
    
    print(f"\n{'='*80}")
    print(f"Evaluation Results - {args.label_type.upper()}")
    print(f"{'='*80}")
    print(f"Macro Dice: {macro_dice:.4f}")
    print(f"\nPer-class Dice:")
    
    class_names = ['Background', 'LV', 'RV', 'LA', 'RA', 'Myocardium', 
                   'Aorta', 'Pulmonary A.', 'SVC', 'IVC']
    
    for c, (name, dice) in enumerate(zip(class_names, per_class_dice)):
        print(f"  Class {c:2d} ({name:15s}): {dice:.4f}")
    
    results = {
        'checkpoint': args.checkpoint,
        'label_type': args.label_type,
        'num_val_cases': len(val_ids),
        'macro_dice': float(macro_dice),
        'per_class_dice': {name: float(dice) for name, dice in zip(class_names, per_class_dice)}
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
