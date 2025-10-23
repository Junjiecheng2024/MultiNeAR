#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================
数据集划分脚本
========================================
文件作用：
    将预处理后的数据集随机划分为训练集和验证集
    
主要功能：
    1. 读取 info.csv（包含所有样本ID）
    2. 随机打乱样本顺序
    3. 按比例划分为训练集和验证集（默认 8:2）
    4. 保存为 info_train.csv 和 info_val.csv

输入：
    - info.csv : 包含所有样本ID的文件

输出：
    - info_train.csv : 训练集样本ID（如 798 个样本）
    - info_val.csv   : 验证集样本ID（如 200 个样本）

使用方法：
    # 默认 8:2 划分
    python split_dataset.py --info_csv /path/to/info.csv
    
    # 自定义比例（如 9:1）
    python split_dataset.py --info_csv /path/to/info.csv --train_ratio 0.9
    
    # 指定随机种子（保证可复现）
    python split_dataset.py --info_csv /path/to/info.csv --seed 42

注意事项：
    ⚠️ 使用固定的随机种子（默认42）保证划分结果可复现
    ⚠️ NeAR 训练时不需要显式的验证集，但划分后便于后续分析
"""

# ============ 导入依赖库 ============
import os                          # 文件系统操作
import pandas as pd                # 数据表格处理
import numpy as np                 # 数值计算和随机数
from pathlib import Path           # 路径处理
import argparse                    # 命令行参数解析


# ============ 核心函数 ============

def split_dataset(info_csv_path, train_ratio=0.8, seed=42, output_dir=None):
    """
    将数据集划分为训练集和验证集
    
    功能：
        读取样本列表，随机打乱并按比例划分
    
    参数：
        info_csv_path: str - info.csv 文件路径（包含列 'id'）
        train_ratio: float - 训练集比例，范围 (0, 1)
                            默认 0.8 表示 80% 训练，20% 验证
        seed: int - 随机种子，用于保证结果可复现
                   默认 42（常用的"答案"种子）
        output_dir: str/None - 输出目录
                              None: 与 info.csv 同目录
                              str: 指定的输出目录
    
    返回：
        train_df: pd.DataFrame - 训练集样本列表
        val_df: pd.DataFrame - 验证集样本列表
    
    用法：
        train_df, val_df = split_dataset("info.csv", train_ratio=0.8, seed=42)
    
    注意：
        ⭐ 使用固定种子保证每次运行结果一致
        ⭐ 训练集和验证集不重叠（互斥）
    """
    # ------- 步骤1：读取数据 -------
    df = pd.read_csv(info_csv_path)                    # 读取 CSV 文件
    print(f"总样本数: {len(df)}")                       # 显示样本总数
    
    # ------- 步骤2：设置随机种子 -------
    np.random.seed(seed)                               # 固定随机种子，保证可复现
    
    # ------- 步骤3：随机打乱样本 -------
    indices = np.random.permutation(len(df))           # 生成随机排列的索引 [0, 1, 2, ..., N-1]
    
    # ------- 步骤4：划分训练集和验证集 -------
    n_train = int(len(df) * train_ratio)               # 计算训练集样本数
    train_indices = indices[:n_train]                  # 前 80% 作为训练集
    val_indices = indices[n_train:]                    # 后 20% 作为验证集
    
    # 根据索引提取对应的行
    train_df = df.iloc[train_indices].reset_index(drop=True)  # reset_index 重新编号 0, 1, 2, ...
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"训练集: {len(train_df)} 样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(val_df)} 样本 ({len(val_df)/len(df)*100:.1f}%)")
    
    # ------- 步骤5：保存文件 -------
    # 确定输出目录（默认与 info.csv 同目录）
    if output_dir is None:
        output_dir = Path(info_csv_path).parent        # 获取父目录
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录（如果不存在）
    
    # 构建输出文件路径
    train_path = output_dir / "info_train.csv"
    val_path = output_dir / "info_val.csv"
    
    # 保存为 CSV（不包含索引列）
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✅ 已保存:")
    print(f"  训练集: {train_path}")
    print(f"  验证集: {val_path}")
    
    return train_df, val_df


# ============ 命令行接口 ============

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="NeAR 数据集划分脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 默认 8:2 划分
    python split_dataset.py --info_csv info.csv
    
    # 自定义 9:1 划分
    python split_dataset.py --info_csv info.csv --train_ratio 0.9
    
    # 指定输出目录
    python split_dataset.py --info_csv info.csv --output_dir ./splits
        """
    )
    
    # 添加命令行参数
    parser.add_argument("--info_csv", type=str, required=True,
                        help="info.csv 文件路径（必需）")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例，范围 (0, 1)，默认 0.8")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，用于保证可复现性，默认 42")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，默认与 info.csv 同目录")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用划分函数
    split_dataset(
        args.info_csv,
        train_ratio=args.train_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )
