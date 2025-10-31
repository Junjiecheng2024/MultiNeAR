"""
Cardiac NeAR V2 Configuration

Optimized for connectivity-aware cardiac segmentation.
Key improvements over V1:
- Increased connectivity loss weight (0.2 → 1.0)
- Unified training/inference resolution (96³)
- Balanced Dice and connectivity objectives
- Extended training (800 epochs)

Expected performance:
- Dice: 0.55-0.58 (slight decrease acceptable)
- Connected Components: 30-50 (target 70% reduction from V1)
"""

cfg = dict(
    # ============ Experiment Settings ============
    run_flag="cardiac_near_v2",
    base_path="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/outputs",
    train_info="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data/info_train.csv",
    val_info="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data/info_val.csv",
    
    # ============ Data Configuration ============
    data_path="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data",
    num_classes=11,  # Background + 10 anatomical structures
    appearance=True,  # Use appearance embeddings
    
    # ============ Resolution Configuration ============
    # Unified resolution to avoid interpolation artifacts during inference
    target_resolution=96,      # Inference resolution
    training_resolution=96,    # Training resolution (must match target)
    grid_noise=0.01,           # Spatial augmentation
    uniform_grid_noise=True,
    
    # ============ Training Hyperparameters ============
    n_epochs=800,              # Extended training for better convergence
    batch_size=8,              # Increased from 6 (due to lower resolution)
    eval_batch_size=4,
    n_training_samples=None,   # Use all training samples
    
    # ============ Optimizer Configuration ============
    lr=3e-4,                   # Learning rate
    weight_decay=1e-4,         # L2 regularization
    patience=150,              # Early stopping patience
    
    # ============ Model Architecture ============
    latent_dim=256,            # Latent code dimension
    
    # ============ Loss Function Configuration ============
    lambda_latent=1e-2,        # Latent code regularization weight
    ignore_index=None,         # No ignored class
    
    # Dice Loss
    lambda_dice=1.0,           # Dice loss weight (reduced from 1.5)
    dice_average="macro",      # Per-class averaging
    label_smoothing=0.05,      # Label smoothing factor
    
    # Connectivity Loss (Key Improvement)
    lambda_conn=1.0,           # Overall connectivity loss weight (5x increase)
    conn_tv_weight=0.3,        # Total Variation: penalizes discontinuities
    conn_boundary_weight=0.15, # Boundary consistency
    conn_compactness_weight=0.15,  # Spatial compactness
    
    # ============ Class Weights ============
    # Rebalanced to emphasize difficult structures
    class_weights=[
        0.5,   # 0: Background
        12.0,  # 1: Myocardium
        12.0,  # 2: LA (Left Atrium)
        12.0,  # 3: LV (Left Ventricle)
        12.0,  # 4: RA (Right Atrium)
        12.0,  # 5: RV (Right Ventricle)
        25.0,  # 6: Aorta
        35.0,  # 7: PA (Pulmonary Artery)
        55.0,  # 8: LAA (Left Atrial Appendage)
        80.0,  # 9: Coronary (most difficult)
        50.0,  # 10: PV (Pulmonary Vein)
    ],
    
    # ============ Scheduler Configuration ============
    scheduler_type="cosine_warmup",  # Cosine annealing with warmup
    warmup_ratio=0.05,               # First 40 epochs (800 * 0.05)
    
    # ============ Training Efficiency ============
    use_amp=True,              # Automatic Mixed Precision
    n_workers=8,               # DataLoader workers
    
    # ============ Logging & Checkpointing ============
    log_interval=10,           # Log every N batches
    save_interval=10,          # Save checkpoint every N epochs
    visualize_interval=50,     # Generate visualizations every N epochs
    seed=42,                   # Random seed for reproducibility
)
