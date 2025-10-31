# near_repair_two_stage.py

"""Two-Stage Training Script for Multi-Class NeAR"""

import os
import sys
import math
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd

# Import your modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from near.datasets.refine_dataset import RefineDataset
from near.models.nn3d.model import EmbeddingDecoder
from near.models.losses import combined_loss
from near.models.nn3d.grid import GatherGridsFromVolumes
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Import config
config_name = os.environ.get("NEAR_CONFIG", "config_cardiac_hybrid")
if config_name == "config_cardiac_two_stage":
    from config_cardiac import cfg_stage1, cfg_stage2
else:
    # Single-stage config
    config_module = __import__(config_name)
    cfg_single = config_module.cfg

def build_scheduler(optimizer, cfg):
    """Build learning rate scheduler with warmup."""
    warmup_epochs = int(0.05 * cfg['n_epochs'])
    total_epochs = cfg['n_epochs']
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def avg_stats(stats_list):
    """Average statistics across batches."""
    if not stats_list:
        return {}
    
    result = {}
    keys = stats_list[0].keys()
    
    for k in keys:
        values = [s[k] for s in stats_list]
        # Check if values are numpy arrays or scalars
        if isinstance(values[0], np.ndarray):
            # Average arrays element-wise
            result[k] = np.mean(values, axis=0)
        else:
            # Average scalars
            result[k] = np.mean(values)
    
    return result

def train_epoch(model, loader, gather_train, optimizer, scheduler, scaler, cfg, device):
    """Train one epoch with grid sampling."""
    model.train()
    stats = []
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        # RefineDataset returns: (index, appearance, seg)
        indices, appearance, seg = batch
        indices = indices.to(device)
        appearance = appearance.to(device)  # [B, 1, D, H, W]
        seg = seg.to(device)  # [B, 1, D, H, W]
        
        # Sample grid points (reduces memory usage)
        _, grid, app_sampled = gather_train(appearance)
        # grid: [B, Ds, Hs, Ws, 3], app_sampled: [B, 1, Ds, Hs, Ws]
        
        # Sample labels at grid points
        labels = torch.nn.functional.grid_sample(
            seg.float(), grid, mode='nearest', align_corners=True
        ).squeeze(1).long()  # [B, Ds, Hs, Ws]
        
        optimizer.zero_grad()
        
        with autocast(enabled=cfg["use_amp"]):
            logits, z = model(indices, grid, app_sampled)
            loss, metrics = combined_loss(
                logits, labels,
                lambda_dice=cfg["lambda_dice"],
                lambda_latent=cfg["lambda_latent"],
                lambda_conn=cfg.get("lambda_conn", 0.0),
                conn_tv_weight=cfg.get("conn_tv_weight", 0.1),
                conn_boundary_weight=cfg.get("conn_boundary_weight", 0.05),
                conn_compactness_weight=cfg.get("conn_compactness_weight", 0.05),
                z=z,
                num_classes=cfg["num_classes"],
                class_weights=torch.tensor(cfg["class_weights"], device=device),
                label_smoothing=cfg["label_smoothing"],
                ignore_index=cfg["ignore_index"],
                dice_average=cfg["dice_average"]
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Handle both scalar and multi-element tensors
        stat_dict = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    stat_dict[k] = v.item()
                else:
                    stat_dict[k] = v.cpu().numpy()  # Keep as array for per-class metrics
            else:
                stat_dict[k] = v
        stats.append(stat_dict)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    scheduler.step()
    return avg_stats(stats)

def evaluate_epoch(model, loader, gather_eval, cfg, device):
    """Evaluate one epoch with grid sampling."""
    model.eval()
    stats = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for batch in pbar:
            # RefineDataset returns: (index, appearance, seg)
            indices, appearance, seg = batch
            indices = indices.to(device)
            appearance = appearance.to(device)  # [B, 1, D, H, W]
            seg = seg.to(device)  # [B, 1, D, H, W]
            
            # Sample grid points (no noise for evaluation)
            _, grid, app_sampled = gather_eval(appearance)
            # grid: [B, D, H, W, 3], app_sampled: [B, 1, D, H, W]
            
            # Sample labels at grid points
            labels = torch.nn.functional.grid_sample(
                seg.float(), grid, mode='nearest', align_corners=True
            ).squeeze(1).long()  # [B, D, H, W]
            
            with autocast(enabled=cfg["use_amp"]):
                logits, z = model(indices, grid, app_sampled)
                loss, metrics = combined_loss(
                    logits, labels,
                    lambda_dice=cfg["lambda_dice"],
                    lambda_latent=cfg["lambda_latent"],
                    lambda_conn=cfg.get("lambda_conn", 0.0),
                    conn_tv_weight=cfg.get("conn_tv_weight", 0.1),
                    conn_boundary_weight=cfg.get("conn_boundary_weight", 0.05),
                    conn_compactness_weight=cfg.get("conn_compactness_weight", 0.05),
                    z=z,
                    num_classes=cfg["num_classes"],
                    class_weights=torch.tensor(cfg["class_weights"], device=device),
                    label_smoothing=cfg["label_smoothing"],
                    ignore_index=cfg["ignore_index"],
                    dice_average=cfg["dice_average"]
                )
            
            # Handle both scalar and multi-element tensors
            stat_dict = {}
            for k, v in metrics.items():
                if torch.is_tensor(v):
                    if v.numel() == 1:
                        stat_dict[k] = v.item()
                    else:
                        stat_dict[k] = v.cpu().numpy()  # Keep as array for per-class metrics
                else:
                    stat_dict[k] = v
            stats.append(stat_dict)
    
    return avg_stats(stats)

def train_stage(stage_name, cfg, model=None, device='cuda'):
    """
    Train one stage.
    
    Args:
        stage_name: "stage1" or "stage2"
        cfg: configuration dict
        model: existing model (for stage2, loaded from stage1)
        device: 'cuda' or 'cpu'
    
    Returns:
        trained model
    """
    print("\n" + "="*80)
    print(f"🚀 Starting {stage_name.upper()}")
    print("="*80)
    print(f"Epochs: {cfg['n_epochs']}")
    print(f"Learning Rate: {cfg['lr']}")
    print(f"Lambda Dice: {cfg['lambda_dice']}")
    print(f"Class Weights: {cfg['class_weights']}")
    print("="*80 + "\n")
    
    # Set seed
    set_seed(42)
    
    # Create output directory
    base_path = os.path.join(cfg["base_path"], cfg["run_flag"])
    os.makedirs(base_path, exist_ok=True)
    
    # Build datasets
    train_set = RefineDataset(
        root=cfg["data_path"],
        info_csv=cfg["train_info"],
        resolution=cfg["target_resolution"],
        n_samples=cfg.get("n_training_samples"),
        augment=True,
        flip_prob=0.5,
        rotate_prob=0.4,
        elastic_prob=0.3,
    )
    
    val_set = RefineDataset(
        root=cfg["data_path"],
        info_csv=cfg["val_info"],
        resolution=cfg["target_resolution"],
        n_samples=None,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["n_workers"],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["n_workers"],
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}\n")
    
    # Build grid samplers
    gather_train = GatherGridsFromVolumes(
        cfg["training_resolution"],  # e.g., 64 or 128
        grid_noise=cfg.get("grid_noise", 0.01),
        uniform_grid_noise=cfg.get("uniform_grid_noise", True)
    )
    
    gather_eval = GatherGridsFromVolumes(
        cfg["target_resolution"],  # e.g., 128
        grid_noise=None  # No noise for evaluation
    )
    print(f"Grid sampling: Train={cfg['training_resolution']}³, Eval={cfg['target_resolution']}³\n")
    
    # Build model (or use existing one for stage2)
    if model is None:
        model = EmbeddingDecoder(
            latent_dimension=cfg["latent_dim"],
            n_samples=len(train_set),
            num_classes=cfg["num_classes"],
            appearance=cfg["appearance"],
            use_fourier=False,
            fourier_bands=6,
            dropout_p=0.1,
        ).to(device)
        print(f"✅ Created new model with {len(train_set)} samples")
    else:
        print(f"✅ Loaded model from stage1")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg["use_amp"])
    
    # TensorBoard
    writer = SummaryWriter(base_path)
    
    # Training loop
    best_dice = 0.0
    no_improve = 0
    
    for epoch in range(1, cfg["n_epochs"] + 1):
        print(f"\n[Epoch {epoch:03d}/{cfg['n_epochs']}]")
        
        # Train
        train_stat = train_epoch(model, train_loader, gather_train, optimizer, scheduler, scaler, cfg, device)
        
        # Validate
        val_stat = evaluate_epoch(model, val_loader, gather_eval, cfg, device)
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        print(f"  LR: {current_lr:.2e}")
        print(f"  Train: loss={train_stat['loss']:.4f} | dice={train_stat['dice_macro']:.4f}")
        print(f"  Val:   loss={val_stat['loss']:.4f} | dice={val_stat['dice_macro']:.4f}")
        
        # TensorBoard
        writer.add_scalar("LR", current_lr, epoch)
        writer.add_scalars("Loss", {"train": train_stat['loss'], "val": val_stat['loss']}, epoch)
        writer.add_scalars("Dice", {"train": train_stat['dice_macro'], "val": val_stat['dice_macro']}, epoch)
        
        # Save best model
        val_dice = val_stat['dice_macro']
        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(base_path, "best.pth"))
            print(f"  ⭐ New best! Dice={best_dice:.4f}")
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= cfg["patience"]:
            print(f"\n⏸ Early stopping at epoch {epoch}. Best Dice={best_dice:.4f}")
            break
    
    writer.close()
    
    print("\n" + "="*80)
    print(f"✅ {stage_name.upper()} Finished!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved: {os.path.join(base_path, 'best.pth')}")
    print("="*80 + "\n")
    
    return model

def main():
    """Single-stage training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*80)
    print("🎯 SINGLE-STAGE TRAINING FOR CARDIAC MULTI-CLASS NEAR")
    print("="*80)
    print(f"Config: {os.environ.get('NEAR_CONFIG', 'config_cardiac_hybrid')}")
    print("="*80 + "\n")
    
    # ==================== SINGLE STAGE ====================
    model = train_stage("single", cfg_single, model=None, device=device)
    
    print("\n" + "="*80)
    print("🎉 SINGLE-STAGE TRAINING COMPLETE!")
    print("="*80)
    print(f"Model: {os.path.join(cfg_single['base_path'], cfg_single['run_flag'], 'best.pth')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()