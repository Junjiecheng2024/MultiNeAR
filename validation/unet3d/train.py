#!/usr/bin/env python3
"""3D U-Net training script with improved strategies."""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
import time
from datetime import datetime
import json
import argparse

# ============ 3D U-Net Model ============
class DoubleConv(nn.Module):
    """(Conv3D -> IN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net with InstanceNorm"""
    def __init__(self, n_channels=1, n_classes=11, base_channels=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        self.outc = nn.Conv3d(base_channels, n_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ============ Dice Loss ============
class DiceLoss(nn.Module):
    """Dice loss that skips empty classes."""
    def __init__(self, n_classes=11, eps=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.eps = eps
    
    def forward(self, logits, target):
        """Compute Dice loss.
        
        Args:
            logits: [B, C, D, H, W]
            target: [B, D, H, W]
        """
        prob = torch.softmax(logits, dim=1)
        dice_list = []
        
        for c in range(self.n_classes):
            pc = prob[:, c]
            tc = (target == c).float()
            
            inter = (pc * tc).sum(dim=(1,2,3))
            pc_sum = pc.sum(dim=(1,2,3))
            tc_sum = tc.sum(dim=(1,2,3))
            denom = pc_sum + tc_sum
            
            mask = (denom > 0)
            if mask.any():
                dice = (2 * inter[mask] + self.eps) / (denom[mask] + self.eps)
                dice_list.append(dice.mean())
        
        if len(dice_list) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return 1 - torch.stack(dice_list).mean()

# ============ Dataset with Foreground Sampling ============
class CardiacDataset(Dataset):
    """Cardiac CT dataset with foreground-aware patch sampling."""
    def __init__(self, image_dir, label_dir, case_ids, patch_size=(96,96,96), 
                 samples_per_epoch=None, fg_ratio=0.7):
        """Initialize dataset.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing labels
            case_ids: List of case IDs (without extension)
            patch_size: Size of training patches
            samples_per_epoch: Number of patches per epoch
            fg_ratio: Probability of sampling from foreground regions
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.case_ids = sorted(list(case_ids))
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch if samples_per_epoch else len(self.case_ids)
        self.fg_ratio = fg_ratio
        
        self.image_paths = {}
        self.label_paths = {}
        
        for cid in self.case_ids:
            self.image_paths[cid] = self._resolve_path(self.image_dir, cid)
            self.label_paths[cid] = self._resolve_path(self.label_dir, cid)
        
        print(f"Dataset initialized with {len(self.case_ids)} cases, "
              f"{self.samples_per_epoch} samples per epoch, fg_ratio={fg_ratio}")
    
    def _resolve_path(self, root, cid):
        """Resolve case path supporting multiple naming formats."""
        for name in [f"{cid}.nii.gz", f"{cid}.nii.img.nii.gz"]:
            p = root / name
            if p.exists():
                return p
        raise FileNotFoundError(f"Case {cid} not found in {root}")
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        cid = np.random.choice(self.case_ids)
        
        image = nib.load(str(self.image_paths[cid])).get_fdata().astype(np.float32)
        label = nib.load(str(self.label_paths[cid])).get_fdata().astype(np.int64)
        
        image = np.clip(image, -1000, 1000)
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        img_patch, lbl_patch = self._sample_patch_fg_aware(image, label, self.fg_ratio)
        
        return torch.from_numpy(img_patch).unsqueeze(0), torch.from_numpy(lbl_patch).long()
    
    def _sample_patch_fg_aware(self, img, lbl, fg_ratio):
        """Sample patch with foreground priority."""
        D, H, W = img.shape
        pd, ph, pw = self.patch_size
        
        want_fg = (np.random.rand() < fg_ratio) and (lbl.max() > 0)
        
        if want_fg:
            fg_coords = np.argwhere(lbl > 0)
            if len(fg_coords) > 0:
                z, y, x = fg_coords[np.random.randint(0, len(fg_coords))]
                d = np.clip(z - pd//2, 0, max(0, D - pd))
                h = np.clip(y - ph//2, 0, max(0, H - ph))
                w = np.clip(x - pw//2, 0, max(0, W - pw))
            else:
                want_fg = False
        
        if not want_fg:
            d = np.random.randint(0, max(1, D - pd + 1))
            h = np.random.randint(0, max(1, H - ph + 1))
            w = np.random.randint(0, max(1, W - pw + 1))
        
        ip = img[d:d+pd, h:h+ph, w:w+pw]
        lp = lbl[d:d+pd, h:h+ph, w:w+pw]
        
        if ip.shape != (pd, ph, pw):
            pad = [(0, pd-ip.shape[0]), (0, ph-ip.shape[1]), (0, pw-ip.shape[2])]
            ip = np.pad(ip, pad, mode='constant', constant_values=0)
            lp = np.pad(lp, pad, mode='constant', constant_values=0)
        
        return ip, lp

# ============ Training Function ============
def train_epoch(model, dataloader, criterion_ce, criterion_dice, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss_ce = criterion_ce(outputs, labels)
        loss_dice = criterion_dice(outputs, labels)
        loss = loss_ce + 0.2 * loss_dice
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}")
            sys.stdout.flush()
    
    return total_loss / num_batches

# ============ Sliding Window Inference ============
def sliding_window_inference(model, volume, patch_size=(96,96,96), stride=(48,48,48), 
                             device='cuda', n_classes=11, batch_size=16):
    """Sliding window inference with batch processing.
    
    Args:
        model: Trained model
        volume: Input volume
        patch_size: Size of sliding window patches
        stride: Stride for sliding window
        device: Device for inference
        n_classes: Number of classes
        batch_size: Batch size for inference
    
    Returns:
        prediction: Predicted segmentation
    """
    model.eval()
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    prob_map = np.zeros((n_classes, D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)
    
    volume_norm = np.clip(volume, -1000, 1000)
    volume_norm = (volume_norm - volume_norm.mean()) / (volume_norm.std() + 1e-8)
    
    z_starts = list(range(0, max(1, D - pd + 1), sd))
    y_starts = list(range(0, max(1, H - ph + 1), sh))
    x_starts = list(range(0, max(1, W - pw + 1), sw))
    
    if z_starts[-1] + pd < D:
        z_starts.append(D - pd)
    if y_starts[-1] + ph < H:
        y_starts.append(H - ph)
    if x_starts[-1] + pw < W:
        x_starts.append(W - pw)
    
    all_coords = []
    for d in z_starts:
        for h in y_starts:
            for w in x_starts:
                all_coords.append((d, h, w))
    
    total_patches = len(all_coords)
    
    with torch.no_grad():
        for batch_start in range(0, total_patches, batch_size):
            batch_end = min(batch_start + batch_size, total_patches)
            batch_coords = all_coords[batch_start:batch_end]
            
            batch_patches = []
            for d, h, w in batch_coords:
                patch = volume_norm[d:d+pd, h:h+ph, w:w+pw]
                
                if patch.shape != (pd, ph, pw):
                    pad = [(0, pd-patch.shape[0]), (0, ph-patch.shape[1]), (0, pw-patch.shape[2])]
                    patch = np.pad(patch, pad, mode='constant', constant_values=0)
                
                batch_patches.append(patch)
            
            batch_tensor = torch.from_numpy(np.array(batch_patches)).float().unsqueeze(1).to(device)
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for idx, (d, h, w) in enumerate(batch_coords):
                prob = probs[idx]
                actual_d = min(pd, D - d)
                actual_h = min(ph, H - h)
                actual_w = min(pw, W - w)
                
                prob_map[:, d:d+actual_d, h:h+actual_h, w:w+actual_w] += prob[:, :actual_d, :actual_h, :actual_w]
                count_map[d:d+actual_d, h:h+actual_h, w:w+actual_w] += 1.0
            
            if batch_end % 100 == 0 or batch_end == total_patches:
                print(f"    Processed {batch_end}/{total_patches} patches", end='\r')
                sys.stdout.flush()
    
    print()
    
    count_map = np.clip(count_map, 1.0, None)
    prob_map /= count_map
    
    prediction = np.argmax(prob_map, axis=0)
    return prediction

# ============ Evaluation Function ============
def evaluate(model, image_dir, label_dir, case_ids, device, num_samples=50):
    """Evaluate model on validation set.
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        label_dir: Directory containing labels
        case_ids: List of case IDs
        device: Device for inference
        num_samples: Number of samples to evaluate
    
    Returns:
        macro_dice: Average Dice score
        dice_per_class: Per-class Dice scores
    """
    model.eval()
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    eval_cases = np.random.choice(case_ids, size=min(num_samples, len(case_ids)), replace=False)
    
    all_dice_per_class = {c: [] for c in range(11)}
    
    print(f"\n  Evaluating on {len(eval_cases)} cases with sliding window...")
    
    for idx, cid in enumerate(eval_cases):
        image_path = None
        label_path = None
        
        for name in [f"{cid}.nii.gz", f"{cid}.nii.img.nii.gz"]:
            p = image_dir / name
            if p.exists():
                image_path = p
                break
        
        for name in [f"{cid}.nii.gz", f"{cid}.nii.img.nii.gz"]:
            p = label_dir / name
            if p.exists():
                label_path = p
                break
        
        if image_path is None or label_path is None:
            continue
        
        image = nib.load(str(image_path)).get_fdata().astype(np.float32)
        label = nib.load(str(label_path)).get_fdata().astype(np.int64)
        
        print(f"  [{idx+1}/{len(eval_cases)}] Evaluating case {cid} ({image.shape})...")
        pred = sliding_window_inference(model, image, patch_size=(96,96,96), 
                                       stride=(48,48,48), device=device, n_classes=11)
        
        for c in range(11):
            pred_c = (pred == c)
            label_c = (label == c)
            
            intersection = np.sum(pred_c & label_c)
            union = np.sum(pred_c) + np.sum(label_c)
            
            if union > 0:
                dice = 2.0 * intersection / union
                all_dice_per_class[c].append(dice)
    
    dice_per_class = {}
    valid_classes = []
    
    for c in range(11):
        if len(all_dice_per_class[c]) > 0:
            dice_per_class[c] = np.mean(all_dice_per_class[c])
            valid_classes.append(dice_per_class[c])
        else:
            dice_per_class[c] = 0.0
    
    macro_dice = np.mean(valid_classes) if valid_classes else 0.0
    
    return macro_dice, dice_per_class

# ============ Train/Val Split ============
def split_train_val(image_dir, val_ratio=0.2, seed=42):
    """Split cases into training and validation sets."""
    image_dir = Path(image_dir)
    
    case_ids = set()
    for p in image_dir.glob("*.nii.gz"):
        stem = p.name.replace(".nii.img.nii.gz", "").replace(".nii.gz", "")
        case_ids.add(stem)
    
    case_ids = sorted(list(case_ids))
    
    np.random.seed(seed)
    n_total = len(case_ids)
    n_val = max(1, int(val_ratio * n_total))
    
    indices = np.random.permutation(n_total)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    val_ids = [case_ids[i] for i in val_indices]
    train_ids = [case_ids[i] for i in train_indices]
    
    return train_ids, val_ids

# ============ Main Function ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_type', type=str, required=True, choices=['baseline', 'repaired'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--eval_epochs', type=str, default=None, help='Specific epochs to evaluate, e.g., "40,50"')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--fg_ratio', type=float, default=0.7)
    args = parser.parse_args()
    
    image_dir = "/home/user/persistent/3dUNet_val/imagesTr"
    
    if args.label_type == 'baseline':
        label_dir = "/home/user/persistent/3dUNet_val/labelsTr_baseline"
    else:
        label_dir = "/home/user/persistent/3dUNet_val/labelsTr_repaired_resized"
    
    output_dir = f"runs_{args.label_type}_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nSplitting dataset with val_ratio={args.val_ratio}, seed=42...")
    train_ids, val_ids = split_train_val(image_dir, val_ratio=args.val_ratio, seed=42)
    print(f"Train cases: {len(train_ids)}, Val cases: {len(val_ids)}")
    
    split_info = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'n_train': len(train_ids),
        'n_val': len(val_ids)
    }
    with open(os.path.join(output_dir, 'train_val_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    model = UNet3D(n_channels=1, n_classes=11, base_channels=32)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    train_dataset = CardiacDataset(image_dir, label_dir, train_ids, 
                                   patch_size=(96, 96, 96), 
                                   samples_per_epoch=len(train_ids)*2,
                                   fg_ratio=args.fg_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True,
                             prefetch_factor=1, persistent_workers=True)
    
    model = model.to(device)
    print(f"Model moved to {device}")
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=11)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    log_file = os.path.join(output_dir, 'training_log.txt')
    
    print(f"\n{'='*80}")
    print(f"Training Configuration - {args.label_type.upper()}")
    print(f"{'='*80}")
    print(f"Label type: {args.label_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    if args.eval_epochs:
        eval_epochs_list = [int(e.strip()) for e in args.eval_epochs.split(',')]
        print(f"Eval epochs: {eval_epochs_list}")
    else:
        eval_epochs_list = None
        print(f"Eval frequency: {args.eval_freq}")
    
    print(f"Val ratio: {args.val_ratio}")
    print(f"Foreground ratio: {args.fg_ratio}")
    print(f"Train cases: {len(train_ids)}")
    print(f"Val cases: {len(val_ids)}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*80}\n")
    
    best_dice = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        sys.stdout.flush()
        
        train_loss = train_epoch(model, train_loader, criterion_ce, criterion_dice, 
                                optimizer, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        log_msg = f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f} - Time: {epoch_time:.1f}s"
        print(log_msg)
        sys.stdout.flush()
        
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        should_eval = False
        if eval_epochs_list:
            should_eval = (epoch in eval_epochs_list)
        else:
            should_eval = (epoch % args.eval_freq == 0 or epoch == args.epochs)
        
        if should_eval:
            print(f"\n  Evaluating on validation set...")
            macro_dice, dice_per_class = evaluate(model, image_dir, label_dir, 
                                                  val_ids, device, num_samples=50)
            
            eval_msg = f"  Val Dice: {macro_dice:.4f}"
            print(eval_msg)
            
            print("  Per-class Dice:")
            for c in range(11):
                if dice_per_class[c] > 0:
                    print(f"    Class {c}: {dice_per_class[c]:.4f}")
            
            with open(log_file, 'a') as f:
                f.write(eval_msg + '\n')
                for c in range(11):
                    if dice_per_class[c] > 0:
                        f.write(f"    Class {c}: {dice_per_class[c]:.4f}\n")
            
            if macro_dice > best_dice:
                best_dice = macro_dice
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"  Best model saved (Dice: {best_dice:.4f})")
                
                with open(os.path.join(output_dir, 'best_dice_per_class.json'), 'w') as f:
                    json.dump(dice_per_class, f, indent=2)
        
        torch.save(model.state_dict(), os.path.join(output_dir, 'latest_model.pth'))
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("Final Evaluation on Validation Set")
    print(f"{'='*80}")
    
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    final_dice, final_dice_per_class = evaluate(model, image_dir, label_dir, 
                                                val_ids, device, num_samples=len(val_ids))
    
    print(f"\nFinal Validation Dice: {final_dice:.4f}")
    print("\nPer-class Dice:")
    for c in range(11):
        if final_dice_per_class[c] > 0:
            print(f"  Class {c}: {final_dice_per_class[c]:.4f}")
    
    final_results = {
        'best_epoch_dice': float(best_dice),
        'final_val_dice': float(final_dice),
        'final_dice_per_class': {str(k): float(v) for k, v in final_dice_per_class.items()},
        'total_time': total_time,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'val_ratio': args.val_ratio,
            'fg_ratio': args.fg_ratio,
            'n_train': len(train_ids),
            'n_val': len(val_ids)
        }
    }
    
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training completed in {total_time/3600:.2f} hours")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Results saved to {output_dir}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
