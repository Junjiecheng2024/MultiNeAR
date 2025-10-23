""""""

Configuration for Cardiac Multi-Class NeAR Training========================================

NeAR 训练配置文件：A6000 生产版本

Hardware: NVIDIA RTX A6000 (49GB VRAM)========================================

Dataset: 998 cardiac CT samples with 11-class segmentation文件作用：

Target: Repair annotations using NeAR (Neural Annotation Refinement)    定义 NeAR 模型训练的所有超参数和配置选项

"""    

使用场景：

cfg = dict(    - 硬件：NVIDIA RTX A6000 49GB GPU

    # Experiment settings    - 数据：998 个心脏 CT 样本（已预处理为 128³）

    run_flag="cardiac_near",    - 目标：11 类心脏器官分割修复

    base_path="runs",

    配置要点：

    # Data paths and resolution    ⭐ target_resolution=128：数据已预处理为 128³，避免动态 resize

    data_path="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data",    ⭐ training_resolution=64：训练时采样 64³ 网格（平衡性能和精度）

    num_classes=11,  # Background + 10 cardiac structures    ⭐ class_weights：高权重给小器官（LAA=403.88，Coronary=687.69）

    appearance=True,  # Use CT intensity features    ⭐ lambda_dice=0.2：Dice loss 权重，关注区域重叠

        

    # Resolution settings使用方法：

    target_resolution=128,      # Data resolution (preprocessed to 128^3)    在 near_repair.py 中导入：

    training_resolution=64,     # Training grid resolution (64^3 for efficiency)    from config_a6000_prod import cfg

    grid_noise=0.01,            # Grid coordinate noise for augmentation    

    uniform_grid_noise=True,    # Use uniform noise distribution修改建议：

        - batch_size：根据 GPU 内存调整（A6000 可用 8-16）

    # Training hyperparameters    - n_epochs：根据收敛情况调整（100 epochs 足够）

    n_epochs=100,    - class_weights：根据实际数据分布微调

    batch_size=8,"""

    eval_batch_size=8,

    n_workers=0,                # DataLoader workers (0 for .npy data)cfg = dict(

    n_training_samples=None,    # Use all 998 samples    # ============ 实验管理配置 ============

        run_flag="cardiac_near_a6000_prod",              # 实验名称标记，用于区分不同实验

    # Optimizer settings                                                     # 格式：<任务>_<硬件>_<版本>

    lr=1e-3,    

    weight_decay=1e-4,    base_path="runs",                                # 实验输出根目录

    milestones=[60, 90],        # LR decay at epochs 60 and 90                                                     # 完整路径：runs/cardiac_near_a6000_prod_<timestamp>/

    gamma=0.1,                  # LR decay factor                                                     # 包含：模型、日志、配置文件

        

    # Model architecture    # ============ 数据与分辨率配置 ============

    latent_dim=256,             # Latent code dimension per sample    data_path="/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data",

        # 数据根目录，包含：

    # Loss function weights    #   - appearance/    : CT 图像（.npy，128×128×128）

    lambda_dice=0.2,            # Dice loss weight    #   - shape/         : 分割掩膜（.npy，128×128×128）

    dice_average="macro",       # Macro-average across classes    #   - info.csv       : 样本ID列表

    lambda_latent=1e-2,         # Latent L2 regularization weight    

    label_smoothing=0.02,       # Label smoothing for CrossEntropy    num_classes=11,                                  # 类别数：0=背景 + 10个心脏器官

    ignore_index=None,          # No ignored class    

        appearance=True,                                 # 是否使用 CT 外观特征（appearance）

    # Class weights (inverse frequency, capped at 100x)                                                     # True: 使用 CT 强度信息（推荐）

    # Order: Background, Myocardium, LA, LV, RA, RV, Aorta, PA, LAA, Coronary, PV                                                     # False: 仅使用形状信息（shape-only）

    class_weights=[    

        1.0000,    # 0: Background    # ------- 分辨率配置（关键性能优化点）-------

        32.2509,   # 1: Myocardium    target_resolution=128,                           # ⭐ 数据的实际分辨率（128³）

        50.1720,   # 2: LA (Left Atrium)                                                     # 数据已在 prepare_near_data.py 中预处理为 128³

        31.3675,   # 3: LV (Left Ventricle)                                                     # 训练时不再需要动态 resize，大幅提升速度

        43.6045,   # 4: RA (Right Atrium)    

        25.9579,   # 5: RV (Right Ventricle)    training_resolution=64,                          # ⭐ 训练时的网格采样分辨率（64³）

        38.0664,   # 6: Aorta                                                     # 从 128³ 数据中随机采样 64³ 网格点

        92.4985,   # 7: PA (Pulmonary Artery)                                                     # 作用：减少计算量，提升训练速度

        100.0,     # 8: LAA (Left Atrial Appendage) - capped                                                     # 注意：推理时使用完整 128³ 分辨率

        100.0,     # 9: Coronary Arteries - capped    

        100.0,     # 10: PV (Pulmonary Veins) - capped    grid_noise=0.01,                                 # 网格坐标加噪声幅度

    ],                                                     # 作用：数据增强，提升模型泛化能力

                                                         # 范围：[0, 0.02]，0.01 是常用值

    # Inference settings    

    model_path="runs/cardiac_near/best.pth",    uniform_grid_noise=True,                         # 是否使用均匀分布噪声

    save_prob=False,            # Save probability maps (False to save space)                                                     # True: 均匀噪声（推荐）

)                                                     # False: 高斯噪声

    
    # ============ 训练超参数配置 ============
    n_epochs=100,                                    # 训练轮数（100 epochs 约 12.4 小时）
                                                     # 经验：60-80 epochs 基本收敛
                                                     # 100 epochs 获得最佳性能
    
    batch_size=8,                                    # 训练批次大小
                                                     # A6000 49GB 可以使用 8（128³ 数据）
                                                     # 显存不足时可降至 4 ⚠️ 已降至4避免OOM
                                                     # 训练速度：约 2.5s/batch
    
    eval_batch_size=8,                               # 验证批次大小（与训练相同即可）
    
    n_workers=0,                                     # DataLoader 工作进程数
                                                     # 0: 主进程加载数据（推荐，数据已预处理为 .npy）
                                                     # >0: 多进程加载（可能引入开销）
    
    n_training_samples=None,                         # 训练样本数量限制
                                                     # None: 使用全部 998 个样本（推荐）
                                                     # int: 仅使用前 N 个样本（用于快速测试）
    
    # ------- 优化器参数 -------
    lr=1e-3,                                         # 初始学习率（Adam 优化器）
                                                     # 1e-3 是常用值，收敛稳定
    
    weight_decay=1e-4,                               # L2 正则化系数
                                                     # 作用：防止过拟合
    
    milestones=[60, 90],                             # 学习率衰减的 epoch 节点
                                                     # 在第 60 和 90 epochs 时降低学习率
    
    gamma=0.1,                                       # 学习率衰减系数
                                                     # 每次衰减时 lr *= gamma
                                                     # 如：1e-3 → 1e-4 → 1e-5
    
    # ============ 模型架构配置 ============
    latent_dim=256,                                  # 潜向量维度
                                                     # EmbeddingDecoder 为每个样本学习一个 256 维向量
                                                     # 作用：捕获样本特异性特征（如器官大小、位置）
                                                     # 经验：128-512 都可行，256 是平衡点
    
    # ============ 损失函数配置 ============
    lambda_dice=0.2,                                 # ⭐ Dice loss 权重
                                                     # 总损失 = CE_loss + 0.2 * Dice_loss + 0.01 * Latent_reg
                                                     # Dice loss 关注区域重叠，对小器官特别重要
    
    dice_average="macro",                            # Dice 计算方式
                                                     # "macro": 每个类别单独计算后取平均（推荐）
                                                     # "micro": 全局计算（受大类别主导）
    
    lambda_latent=1e-2,                              # 潜向量正则化权重
                                                     # 作用：约束潜向量不要过大，防止过拟合
    
    label_smoothing=0.02,                            # 标签平滑系数（CrossEntropy loss）
                                                     # 作用：软化 one-hot 标签，提升泛化能力
                                                     # 范围：[0, 0.1]，0.02 是轻度平滑
    
    ignore_index=None,                               # 忽略的类别索引（None 表示不忽略任何类）
                                                     # 用途：如果某些类别标注不可靠，可设为 -1 等
    
    # ============ 类别权重配置 ============
    # ⭐ 根据类别分布计算的反频率权重
    # 作用：平衡类别不平衡，给小器官（LAA、Coronary）更高权重
    # 计算方式：weight[c] = total_voxels / (num_classes * class_voxels[c])
    # 
    # 🔧 修改日期：2025-10-19
    # 🔧 修改原因：原始权重过高导致模型过度预测 Class 8/9，忽略背景
    #    - Class 9 原权重 687.69 → 限制为 100
    #    - Class 8 原权重 403.88 → 限制为 100
    #    - Class 10 原权重 165.20 → 限制为 100
    # 🔧 预期效果：背景 Dice 从 0.012 提升到 >0.70
    class_weights=[
        1.0000,    # 0: Background（背景，归一化基准）
        32.2509,   # 1: Myocardium（心肌，中等大小）
        50.1720,   # 2: LA（左心房，大器官）
        31.3675,   # 3: LV（左心室，最大器官）
        43.6045,   # 4: RA（右心房，大器官）
        25.9579,   # 5: RV（右心室，大器官）
        38.0664,   # 6: Aorta（主动脉，中等大小）
        92.4985,   # 7: PA（肺动脉，较小）
        100.0,     # 8: LAA（左心耳）✅ 修改：限制最大权重 100x（原: 403.88）
        100.0,     # 9: Coronary（冠状动脉）✅ 修改：限制最大权重 100x（原: 687.69）
        100.0,     # 10: PV（肺静脉）✅ 修改：限制最大权重 100x（原: 165.20）
    ],
    # 注意：
    #   - 限制最大权重为 100x，防止过度关注某个类别
    #   - 仍然给小结构更多关注，但不至于导致背景预测崩溃
    #   - 大器官（LV/LA/RV）权重保持不变（25-50）
    
    # ============ 推理配置 ============
    model_path="runs/cardiac_near_a6000_prod/best.pth",
    # 最佳模型路径（用于推理）
    # 注意：实际运行时会动态更新为最新的 best.pth 路径
    
    save_prob=False,                                 # 是否保存概率图（softmax输出）
                                                     # False: 仅保存预测标签（节省空间）
                                                     # True: 保存所有类别的概率（用于不确定性分析）
)

