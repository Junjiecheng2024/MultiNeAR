#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证脚本：快速检查预处理后的数据是否正确
================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from near.datasets.refine_dataset import CardiacMultiClassDataset


def validate_dataset(data_root, num_samples_to_check=5):
    """验证数据集"""
    print("=" * 80)
    print("数据集验证")
    print("=" * 80)
    
    # 检查文件结构
    data_root = Path(data_root)
    print(f"\n数据根目录: {data_root}")
    
    required_dirs = ["appearance", "shape"]
    required_files = ["info.csv", "class_statistics.csv", "class_weights.json"]
    
    print("\n检查文件结构:")
    for d in required_dirs:
        exists = (data_root / d).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {d}/")
    
    for f in required_files:
        exists = (data_root / f).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {f}")
    
    # 加载数据集
    print("\n加载数据集...")
    try:
        dataset = CardiacMultiClassDataset(
            root=str(data_root),
            resolution=None,  # 保持原分辨率
            normalize=True
        )
        print(f"  ✓ 数据集加载成功: {len(dataset)} 个样本")
    except Exception as e:
        print(f"  ✗ 数据集加载失败: {e}")
        return
    
    # 检查类别统计
    print("\n类别统计:")
    stats_path = data_root / "class_statistics.csv"
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        print(stats_df.to_string(index=False))
    
    # 随机检查几个样本
    print(f"\n随机检查 {num_samples_to_check} 个样本:")
    indices = np.random.choice(len(dataset), min(num_samples_to_check, len(dataset)), replace=False)
    
    for idx in indices:
        try:
            _, app, seg = dataset[idx]
            case_id = dataset.info.loc[idx, dataset.id_key]
            
            # 检查形状
            app_shape = app.shape  # [1, D, H, W]
            seg_shape = seg.shape  # [1, D, H, W]
            
            # 检查值域
            app_min, app_max = app.min().item(), app.max().item()
            seg_min, seg_max = seg.min().item(), seg.max().item()
            seg_unique = np.unique(seg.numpy())
            
            print(f"\n  样本 {idx} (ID: {case_id}):")
            print(f"    - Appearance shape: {app_shape}, 值域: [{app_min:.4f}, {app_max:.4f}]")
            print(f"    - Segmentation shape: {seg_shape}, 值域: [{seg_min}, {seg_max}]")
            print(f"    - 包含的类别: {seg_unique.tolist()}")
            
            # 检查是否有异常值
            if seg_max >= 11:
                print(f"    ⚠ 警告: 分割包含超出范围的类别 {seg_max}")
            if seg_min < 0:
                print(f"    ⚠ 警告: 分割包含负值 {seg_min}")
            
        except Exception as e:
            print(f"\n  样本 {idx} 检查失败: {e}")
    
    # 可视化一个样本
    print("\n生成可视化...")
    try:
        idx = 0
        _, app, seg = dataset[idx]
        case_id = dataset.info.loc[idx, dataset.id_key]
        
        app_np = app.numpy()[0]  # [D, H, W]
        seg_np = seg.numpy()[0]  # [D, H, W]
        
        # 中间切片
        mid_slice = app_np.shape[0] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # CT 图像
        axes[0].imshow(app_np[mid_slice], cmap='gray')
        axes[0].set_title(f"CT Image (Case {case_id})")
        axes[0].axis('off')
        
        # 分割（overlay）
        axes[1].imshow(app_np[mid_slice], cmap='gray', alpha=0.7)
        axes[1].imshow(seg_np[mid_slice], cmap='tab20', alpha=0.5, vmin=0, vmax=10)
        axes[1].set_title("Segmentation Overlay")
        axes[1].axis('off')
        
        # 分割（单独）
        axes[2].imshow(seg_np[mid_slice], cmap='tab20', vmin=0, vmax=10)
        axes[2].set_title("Segmentation Only")
        axes[2].axis('off')
        
        # 添加图例
        class_names = [
            "Background", "Myocardium", "LA", "LV", "RA", "RV",
            "Aorta", "PA", "LAA", "Coronary", "PV"
        ]
        legend_text = "\n".join([f"{i}: {name}" for i, name in enumerate(class_names)])
        fig.text(1.02, 0.5, legend_text, fontsize=9, verticalalignment='center')
        
        plt.tight_layout()
        
        vis_path = data_root / "dataset_validation.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 可视化保存至: {vis_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"  ✗ 可视化失败: {e}")
    
    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)
    print("\n如果所有检查通过，可以开始训练:")
    print("  cd repairing/near_repairing")
    print("  python3 near_repair.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集验证脚本")
    parser.add_argument("--data_root", type=str, required=True,
                        help="预处理后的数据根目录")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="检查的样本数")
    
    args = parser.parse_args()
    
    validate_dataset(args.data_root, args.num_samples)
