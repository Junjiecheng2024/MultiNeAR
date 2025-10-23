"""Configuration for Cardiac Multi-Class NeAR Training

Hardware: NVIDIA RTX A6000 (49GB VRAM)
Dataset: 998 cardiac CT samples with 11-class segmentation
Target: Repair annotations using NeAR (Neural Annotation Refinement)
"""

cfg = dict(
    # ============ Experiment Settings ============
    run_flag="cardiac_near_a6000_prod",
    base_path="runs",
    
    # ============ Data Configuration ============
    data_path="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data",
    num_classes=11,
    appearance=True,
    
    # ============ Resolution Configuration ============
    target_resolution=128,
    training_resolution=64,
    grid_noise=0.01,
    uniform_grid_noise=True,
    
    # ============ Training Hyperparameters ============
    n_epochs=100,
    batch_size=4,
    eval_batch_size=8,
    n_workers=0,
    n_training_samples=None,
    
    # ============ Optimizer Configuration ============
    lr=1e-3,
    weight_decay=1e-4,
    milestones=[60, 90],
    gamma=0.1,
    
    # ============ Model Architecture ============
    latent_dim=256,
    
    # ============ Loss Function Configuration ============
    lambda_dice=0.2,
    dice_average="macro",
    lambda_latent=1e-2,
    label_smoothing=0.02,
    ignore_index=None,
    
    # ============ Class Weights ============
    # Order: Background, Myocardium, LA, LV, RA, RV, Aorta, PA, LAA, Coronary, PV
    # Computed as inverse frequency, capped at 100x
    class_weights=[
        1.0000,     # 0: Background
        32.2509,    # 1: Myocardium
        50.1720,    # 2: LA (Left Atrium)
        31.3675,    # 3: LV (Left Ventricle)
        43.6045,    # 4: RA (Right Atrium)
        25.9579,    # 5: RV (Right Ventricle)
        38.0664,    # 6: Aorta
        92.4985,    # 7: PA (Pulmonary Artery)
        100.0,      # 8: LAA (Left Atrial Appendage) - capped
        100.0,      # 9: Coronary Arteries - capped
        100.0,      # 10: PV (Pulmonary Veins) - capped
    ],
    
    # ============ Inference Configuration ============
    model_path="runs/cardiac_near/best.pth",
    save_prob=False,
)
