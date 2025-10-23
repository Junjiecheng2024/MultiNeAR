import _init_paths_local

import os
import time
import importlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from near.datasets.refine_dataset import CardiacMultiClassDataset
from near.models.nn3d.grid import GatherGridsFromVolumes
from near.models.nn3d.model import EmbeddingDecoder
from near.models.losses import combined_loss


def setup_cfg(cfg):
    cfg["run_flag"] += time.strftime("_%y%m%d_%H%M%S")
    base_path = os.path.join(cfg["base_path"], cfg["run_flag"])
    os.makedirs(base_path, exist_ok=False)
    
    try:
        import json
        config_path = os.path.join(base_path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    
    return cfg, base_path


def to_device(x, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(xx, device) for xx in x)
    
    return x.to(device) if hasattr(x, "to") else x


def avg_stats(stats_list):
    out = {}
    n = len(stats_list) if stats_list else 1
    
    for s in stats_list:
        for k, v in s.items():
            out[k] = out.get(k, 0.0) + float(v)
    
    for k in out:
        out[k] /= n
    
    return out


@torch.no_grad()
def _sample_labels_nearest(seg_long, grids):
    """ Sample labels from segmentation at grid points using nearest neighbor.
    
    Args:
        seg_long: [B,1,D,H,W] segmentation labels (integer 0~K-1)
        grids: [B,Ds,Hs,Ws,3] sampling coordinates in [-1,1]^3
    
    Returns:
        labels: [B,Ds,Hs,Ws] sampled labels (long)
    """
    lab = F.grid_sample(
        seg_long.float(),
        grids,
        mode='nearest',
        align_corners=True
    )
    return lab.squeeze(1).long()


def train_one_epoch(model, loader, gather_train, optimizer, cfg, device, class_weights_t=None):
    """ Train for one epoch. """
    model.train()
    stats_epoch = []

    for (idx_tensor, app, seg) in tqdm(loader, desc="  Train", leave=False):
        idx_tensor = to_device(idx_tensor, device)
        app = to_device(app, device)
        seg = to_device(seg, device)

        _, grids, app_label = gather_train(app)
        labels = _sample_labels_nearest(seg, grids)

        logits, z = model(idx_tensor, grids, app_label)

        loss, stat = combined_loss(
            logits=logits,
            labels=labels,
            lambda_dice=cfg.get("lambda_dice", 0.0),
            lambda_latent=cfg.get("lambda_latent", 0.0),
            z=z,
            num_classes=cfg["num_classes"],
            class_weights=class_weights_t,
            label_smoothing=cfg.get("label_smoothing", 0.0),
            ignore_index=cfg.get("ignore_index", None),
            dice_average=cfg.get("dice_average", "macro")
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        stat_dict = {}
        for k, v in stat.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    stat_dict[k] = v.item()
                else:
                    stat_dict[k] = v.mean().item()
            else:
                stat_dict[k] = float(v)
        stats_epoch.append(stat_dict)

    return avg_stats(stats_epoch)


@torch.no_grad()
def eval_one_epoch(model, loader, gather_eval, cfg, device, class_weights_t=None):
    """ Evaluate for one epoch. """
    model.eval()
    stats_epoch = []

    for (idx_tensor, app, seg) in tqdm(loader, desc="  Eval", leave=False):
        idx_tensor = to_device(idx_tensor, device)
        app = to_device(app, device)
        seg = to_device(seg, device)

        _, grids, app_label = gather_eval(app)
        labels = _sample_labels_nearest(seg, grids)

        logits, z = model(idx_tensor, grids, app_label)

        loss, stat = combined_loss(
            logits=logits,
            labels=labels,
            lambda_dice=cfg.get("lambda_dice", 0.0),
            lambda_latent=cfg.get("lambda_latent", 0.0),
            z=z,
            num_classes=cfg["num_classes"],
            class_weights=class_weights_t,
            label_smoothing=cfg.get("label_smoothing", 0.0),
            ignore_index=cfg.get("ignore_index", None),
            dice_average=cfg.get("dice_average", "macro")
        )

        stat_dict = {}
        for k, v in stat.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    stat_dict[k] = v.item()
                else:
                    stat_dict[k] = v.mean().item()
            else:
                stat_dict[k] = float(v)
        stats_epoch.append(stat_dict)

    return avg_stats(stats_epoch)


def main():
    # Load configuration
    config_name = os.environ.get("NEAR_CONFIG", "config_cardiac")
    cfg_module = importlib.import_module(config_name)
    print(f"Using config: {config_name}.py")
    
    cfg, base_path = setup_cfg(cfg_module.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare datasets
    train_set = CardiacMultiClassDataset(
        root=cfg["data_path"],
        resolution=cfg["target_resolution"],
        n_samples=cfg.get("n_training_samples", None),
        normalize=True
    )
    
    eval_set = CardiacMultiClassDataset(
        root=cfg["data_path"],
        resolution=cfg["target_resolution"],
        n_samples=cfg.get("n_training_samples", None),
        normalize=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["n_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    print(f"Train set: {len(train_set)} samples, batch_size={cfg['batch_size']}")
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["n_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    print(f"Eval set: {len(eval_set)} samples, batch_size={cfg['eval_batch_size']}")

    # Initialize grid samplers
    gather_train = GatherGridsFromVolumes(
        cfg["training_resolution"],
        grid_noise=cfg.get("grid_noise", 0.0),
        uniform_grid_noise=cfg.get("uniform_grid_noise", True)
    )
    
    gather_eval = GatherGridsFromVolumes(
        cfg["target_resolution"],
        grid_noise=None
    )

    # Initialize model
    model = EmbeddingDecoder(
        latent_dimension=cfg.get("latent_dim", 256),
        n_samples=len(train_set),
        num_classes=cfg["num_classes"],
        appearance=cfg.get("appearance", True)
    )
    
    model = to_device(model, device)
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.get("milestones", []),
        gamma=cfg.get("gamma", 0.1)
    )

    # Prepare class weights
    class_weights = cfg.get("class_weights", None)
    class_weights_t = None
    if class_weights is not None:
        class_weights_t = to_device(torch.tensor(class_weights, dtype=torch.float32), device)
    
    print("=" * 80)
    print("Starting training...")
    print(f"Model: EmbeddingDecoder (latent_dim={cfg.get('latent_dim', 256)}, "
          f"n_samples={len(train_set)}, num_classes={cfg['num_classes']})")
    print(f"Optimizer: Adam (lr={cfg['lr']})")
    print(f"Device: {device}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Multi-GPU: {torch.cuda.device_count()} GPUs")
    print("=" * 80)

    # Epoch 0 baseline evaluation
    print("\nEpoch 0 baseline evaluation...")
    best_val = None
    
    eval_stat = eval_one_epoch(model, eval_loader, gather_eval, cfg, device, class_weights_t)
    print(f"Epoch 0 evaluation completed: {eval_stat}")
    
    torch.save(model.state_dict(), os.path.join(base_path, "latest.pth"))
    torch.save(model.state_dict(), os.path.join(base_path, "best.pth"))
    
    best_val = eval_stat["loss"]
    print(f"[Eval@0] loss={eval_stat['loss']:.4f}, ce={eval_stat['ce']:.4f}, "
          f"dice_macro={eval_stat['dice_macro']:.4f}")

    # Training loop
    print("\nStarting training loop...")
    for epoch in range(1, cfg["n_epochs"] + 1):
        train_stat = train_one_epoch(
            model, train_loader, gather_train, optimizer, cfg, device, class_weights_t
        )
        
        eval_stat = eval_one_epoch(
            model, eval_loader, gather_eval, cfg, device, class_weights_t
        )

        # Save checkpoints
        torch.save(model.state_dict(), os.path.join(base_path, "latest.pth"))

        if (best_val is None) or (eval_stat["loss"] < best_val):
            best_val = eval_stat["loss"]
            torch.save(model.state_dict(), os.path.join(base_path, "best.pth"))
            
            print("=" * 72)
            print(f"[BEST @ epoch {epoch}] val_loss={best_val:.4f}")
            print("=" * 72)

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_stat['loss']:.4f} | ce={train_stat['ce']:.4f} | "
              f"dice_macro={train_stat['dice_macro']:.4f} || "
              f"val_loss={eval_stat['loss']:.4f} | ce={eval_stat['ce']:.4f} | "
              f"dice_macro={eval_stat['dice_macro']:.4f}")

        scheduler.step()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Output directory: {base_path}")
    print(f"Latest model: {os.path.join(base_path, 'latest.pth')}")
    print(f"Best model: {os.path.join(base_path, 'best.pth')} (val_loss={best_val:.4f})")
    print(f"Config backup: {os.path.join(base_path, 'config.json')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
