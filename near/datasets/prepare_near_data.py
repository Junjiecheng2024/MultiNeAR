#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================
NeAR 数据预处理脚本
========================================
文件作用：
    将原始的 NIfTI 格式(.nii.gz)心脏 CT 数据转换为 NeAR 训练所需的 NumPy 格式(.npy)
    
主要功能：
    1. 读取 .nii.gz 格式的 CT 图像和分割掩膜
    2. 统一重采样到固定分辨率（128³，消除动态 resize 瓶颈）
    3. HU 值裁剪（-1000~1000）但不归一化（留给 Dataset 动态处理）
    4. 保存为 .npy 格式：
       - appearance/ : CT 图像（HU 值）
       - shape/      : 分割掩膜（11类标签）
    5. 生成 info.csv 样本列表
    6. 计算类别分布统计（用于生成类别权重）

输入：
    - images/        : 原始 CT 图像（.nii.gz）
    - segmentations/ : 分割掩膜（.nii.gz，11类）

输出：
    - appearance/           : CT 图像 NumPy 数组（128×128×128，float32）
    - shape/                : 分割掩膜 NumPy 数组（128×128×128，int16）
    - info.csv              : 样本ID列表
    - class_statistics.csv  : 类别分布统计
    - class_weights.json    : 推荐的类别权重

使用方法：
    # 使用默认配置
    python prepare_near_data.py
    
    # 自定义参数
    python prepare_near_data.py --image_dir /path/to/images \
                                 --seg_dir /path/to/segs \
                                 --output_root /path/to/output \
                                 --n_workers 8

性能：
    - 多进程处理：默认使用 CPU 核心数的一半
    - 处理速度：约 1-2 秒/样本（取决于原始尺寸）
    - 内存需求：约 2GB/进程

注意事项：
    ⚠️ 重要：这里只裁剪 HU 值，不做归一化！
              归一化留给 Dataset 在训练时动态完成，这样更灵活
    ⚠️ 输出分辨率固定为 128³，这是训练时的关键优化点
"""

# ============ 导入依赖库 ============
import os                          # 文件系统操作
import sys                         # 系统相关
import argparse                    # 命令行参数解析
import numpy as np                 # 数值计算
import pandas as pd                # 数据表格处理
import SimpleITK as sitk           # 医学图像读取和重采样
from pathlib import Path           # 路径处理（更现代的方式）
from tqdm import tqdm              # 进度条显示
from multiprocessing import Pool, cpu_count  # 多进程并行处理
from skimage.transform import resize as sk_resize  # 图像缩放（更快）
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息


# ============ 配置参数 ============
"""
默认配置字典
-----------
用法：根据实际数据路径修改 image_dir 和 seg_dir
"""

DEFAULT_CONFIG = {
    # -------- 输入路径 --------
    "image_dir": "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/matched_dataset/images",
    # CT 图像目录，包含 *.nii.img.nii.gz 文件
    
    "seg_dir": "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/matched_dataset/segmentations",
    # 分割掩膜目录，包含 *.nii.img.nii.gz 文件，标签值 0-10
    
    # -------- 输出路径 --------
    "output_root": "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data",
    # 输出根目录，将创建 appearance/ 和 shape/ 子目录
    
    # -------- 处理参数 --------
    "target_resolution": (128, 128, 128),  
    # ⭐ 关键优化：预先 resize 到 128³，避免训练时动态 resize 导致的瓶颈
    # 原版 NeAR 默认使用 128³，这里保持一致
    
    "hu_min": -1000,  
    # HU 值下限（空气的 HU 值约为 -1000）
    
    "hu_max": 1000,   
    # HU 值上限（骨骼的 HU 值约为 +1000）
    
    "normalize_method": "minmax",  
    # ⚠️ 注意：这个参数实际上不再使用
    # 保存时只裁剪 HU 值，不做归一化
    # 归一化留给 Dataset.__getitem__() 动态完成
    
    # -------- 类别信息 --------
    "num_classes": 11,
    # 11 类心脏器官：0=背景, 1=心肌, 2=左心房, 3=左心室, 4=右心房, 5=右心室,
    #               6=主动脉, 7=肺动脉, 8=左心耳, 9=冠状动脉, 10=肺静脉
    
    "class_names": [
        "Background",   # 0: 背景
        "Myocardium",   # 1: 心肌
        "LA",           # 2: 左心房（Left Atrium）
        "LV",           # 3: 左心室（Left Ventricle）
        "RA",           # 4: 右心房（Right Atrium）
        "RV",           # 5: 右心室（Right Ventricle）
        "Aorta",        # 6: 主动脉
        "PA",           # 7: 肺动脉（Pulmonary Artery）
        "LAA",          # 8: 左心耳（Left Atrial Appendage）⭐ 重点关注
        "Coronary",     # 9: 冠状动脉
        "PV"            # 10: 肺静脉（Pulmonary Vein）
    ],
    
    # -------- 多进程配置 --------
    "n_workers": min(8, cpu_count() // 2),
    # 并行处理进程数：默认使用 CPU 核心数的一半，最多 8 个
    # 增加进程数可以加速处理，但会增加内存消耗
}


# ============ 工具函数 ============
"""
基础工具函数：文件加载、重采样、归一化等
"""

def load_nifti(path):
    """
    加载 NIfTI 格式医学图像
    
    功能：
        使用 SimpleITK 读取 .nii.gz 文件，提取数组和元数据
    
    参数：
        path: str/Path - NIfTI 文件路径
    
    返回：
        arr: np.ndarray - 图像数组，shape=(D, H, W)，D=深度，H=高度，W=宽度
        spacing: tuple - 体素间距(x, y, z)，单位 mm
        img: sitk.Image - SimpleITK 图像对象（包含完整元数据）
    
    用法：
        ct_array, spacing, img = load_nifti("patient001.nii.gz")
    """
    img = sitk.ReadImage(str(path))                    # 读取 NIfTI 文件
    arr = sitk.GetArrayFromImage(img)                  # 转换为 numpy 数组，shape=[D, H, W]
    spacing = img.GetSpacing()                         # 获取体素间距 (x, y, z) mm
    return arr, spacing, img


def resample_volume(volume, original_spacing, target_spacing, is_label=False):
    """
    重采样 3D 体积到目标分辨率
    
    功能：
        将不同尺寸的医学图像统一到固定分辨率（如 128³）
        支持两种模式：
        1. 基于目标体素数的 resize（target_spacing 值 ≥ 10，如 128）
        2. 基于物理间距的 resample（target_spacing 值 < 10，如 1.0mm）
    
    参数：
        volume: np.ndarray - 输入 3D 数组，shape=(D, H, W)
        original_spacing: tuple - 原始体素间距 (x, y, z) mm
        target_spacing: tuple - 目标分辨率/间距
                               如果值≥10: 视为目标体素数，如 (128, 128, 128)
                               如果值<10: 视为物理间距，如 (1.0, 1.0, 1.0) mm
        is_label: bool - 是否为分割标签
                        True: 使用最近邻插值（保持整数标签）
                        False: 使用三次插值（平滑的CT图像）
    
    返回：
        resampled: np.ndarray - 重采样后的 3D 数组
    
    用法：
        # 模式1：Resize 到 128×128×128 体素
        ct_128 = resample_volume(ct, spacing, (128, 128, 128), is_label=False)
        
        # 模式2：Resample 到 1mm 各向同性
        ct_iso = resample_volume(ct, spacing, (1.0, 1.0, 1.0), is_label=False)
    
    注意：
        ⭐ 本项目使用模式1（Resize到128³），这是关键的性能优化点！
           在预处理时统一尺寸，训练时避免动态 resize 瓶颈
    """
    # 判断是哪种模式：如果 target_spacing 的值都很大（≥10），则视为目标体素数
    if all(s >= 10 for s in target_spacing):
        # ------- 模式1：直接 Resize 到目标体素数 -------
        target_shape = tuple(int(s) for s in target_spacing)  # 转换为整数 tuple
        
        if is_label:
            # 标签使用最近邻插值（order=0），保持整数值
            resampled = sk_resize(
                volume, 
                target_shape, 
                order=0,              # 最近邻插值（不引入新的标签值）
                preserve_range=True,  # 保持原始数值范围
                anti_aliasing=False   # 不使用抗锯齿（标签不需要）
            ).astype(volume.dtype)    # 保持原始数据类型
        else:
            # CT 图像使用三次插值（order=3），更平滑
            resampled = sk_resize(
                volume, 
                target_shape, 
                order=3,              # 三次插值（高质量）
                preserve_range=True   # 保持 HU 值范围
            ).astype(np.float32)      # 转换为 float32
        
        return resampled
    
    else:
        # ------- 模式2：基于物理间距的 Resample -------
        # 使用 SimpleITK 进行精确的物理空间重采样
        img = sitk.GetImageFromArray(volume)
        img.SetSpacing(original_spacing)
        
        # 计算新尺寸
        original_size = img.GetSize()
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
        ]
        
        # 选择插值方法
        interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
        
        # 重采样
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetDefaultPixelValue(0)
        
        resampled = resampler.Execute(img)
        return sitk.GetArrayFromImage(resampled)


def normalize_ct(ct_array, hu_min=-1000, hu_max=1000, method="minmax"):
    """归一化 CT 图像
    
    注意：保存时不做归一化，保存原始 HU 值（裁剪后）
    归一化留给 Dataset 在加载时动态完成，这样更灵活
    """
    # 裁剪 HU 值到合理范围
    ct_clipped = np.clip(ct_array, hu_min, hu_max)
    
    # 直接返回裁剪后的 HU 值，不做归一化
    # Dataset 会在 __getitem__ 中根据 normalize 参数决定是否归一化
    return ct_clipped.astype(np.float32)


def validate_segmentation(seg_array, num_classes=11):
    """验证分割掩膜的值域"""
    unique_values = np.unique(seg_array)
    if np.max(unique_values) >= num_classes or np.min(unique_values) < 0:
        return False, unique_values
    return True, unique_values


def compute_class_statistics(seg_array, num_classes=11):
    """计算每个类别的体素数"""
    stats = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        stats[c] = np.sum(seg_array == c)
    return stats


# ============ 单样本处理函数 ============

def process_single_case(args):
    """处理单个样本（用于多进程）"""
    case_id, config = args
    
    try:
        # 构建文件路径
        img_path = Path(config["image_dir"]) / f"{case_id}.nii.img.nii.gz"
        seg_path = Path(config["seg_dir"]) / f"{case_id}.nii.img.nii.gz"
        
        # 检查文件是否存在
        if not img_path.exists():
            return {"case_id": case_id, "status": "missing_image", "error": str(img_path)}
        if not seg_path.exists():
            return {"case_id": case_id, "status": "missing_seg", "error": str(seg_path)}
        
        # 加载数据
        ct_array, ct_spacing, ct_img = load_nifti(img_path)
        seg_array, seg_spacing, seg_img = load_nifti(seg_path)
        
        # 验证分割掩膜
        is_valid, unique_vals = validate_segmentation(seg_array, config["num_classes"])
        if not is_valid:
            return {
                "case_id": case_id,
                "status": "invalid_seg",
                "error": f"Invalid labels: {unique_vals}"
            }
        
        # 可选：重采样
        if config["target_resolution"] is not None:
            target_spacing = config["target_resolution"]
            ct_array = resample_volume(ct_array, ct_spacing, target_spacing, is_label=False)
            seg_array = resample_volume(seg_array, seg_spacing, target_spacing, is_label=True)
        
        # 裁剪 HU 值（但不归一化，归一化留给 Dataset）
        ct_clipped = normalize_ct(
            ct_array,
            config["hu_min"],
            config["hu_max"],
            config["normalize_method"]  # 这个参数现在不起作用了
        )
        
        # 转换分割为整型
        seg_int = seg_array.astype(np.int16)
        
        # 计算类别统计
        class_stats = compute_class_statistics(seg_int, config["num_classes"])
        
        # 保存为 .npy
        app_out_path = Path(config["output_root"]) / "appearance" / f"{case_id}.npy"
        seg_out_path = Path(config["output_root"]) / "shape" / f"{case_id}.npy"
        
        app_out_path.parent.mkdir(parents=True, exist_ok=True)
        seg_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(app_out_path, ct_clipped)
        np.save(seg_out_path, seg_int)
        
        return {
            "case_id": case_id,
            "status": "success",
            "original_shape": ct_array.shape,
            "original_spacing": ct_spacing,
            "class_stats": class_stats
        }
        
    except Exception as e:
        return {
            "case_id": case_id,
            "status": "error",
            "error": str(e)
        }


# ============ 主处理流程 ============

def get_case_ids(image_dir):
    """从图像目录获取所有 case ID"""
    image_dir = Path(image_dir)
    case_ids = []
    
    # 匹配 *.nii.img.nii.gz 格式
    for f in sorted(image_dir.glob("*.nii.img.nii.gz")):
        # 从 "1.nii.img.nii.gz" 提取 "1"
        case_id = f.name.replace(".nii.img.nii.gz", "")
        case_ids.append(case_id)
    
    return case_ids


def main(config):
    """主处理函数"""
    print("=" * 80)
    print("NeAR 数据预处理脚本")
    print("=" * 80)
    
    # 创建输出目录
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "appearance").mkdir(exist_ok=True)
    (output_root / "shape").mkdir(exist_ok=True)
    
    # 获取所有样本 ID
    case_ids = get_case_ids(config["image_dir"])
    print(f"\n发现 {len(case_ids)} 个样本")
    print(f"输出目录: {output_root}")
    print(f"使用 {config['n_workers']} 个进程\n")
    
    # 准备多进程参数
    process_args = [(cid, config) for cid in case_ids]
    
    # 多进程处理
    print("开始处理...")
    results = []
    
    if config["n_workers"] > 1:
        with Pool(config["n_workers"]) as pool:
            for result in tqdm(pool.imap(process_single_case, process_args), total=len(case_ids)):
                results.append(result)
    else:
        for args in tqdm(process_args):
            results.append(process_single_case(args))
    
    # 统计结果
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_results = [r for r in results if r["status"] != "success"]
    
    print(f"\n处理完成:")
    print(f"  成功: {success_count}/{len(case_ids)}")
    print(f"  失败: {len(failed_results)}/{len(case_ids)}")
    
    if failed_results:
        print("\n失败样本:")
        for r in failed_results[:10]:  # 只显示前10个
            print(f"  - {r['case_id']}: {r['status']} ({r.get('error', '')})")
        if len(failed_results) > 10:
            print(f"  ... 还有 {len(failed_results) - 10} 个失败样本")
    
    # 生成 info.csv
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        info_df = pd.DataFrame([
            {"id": r["case_id"]}
            for r in successful_results
        ])
        info_path = output_root / "info.csv"
        info_df.to_csv(info_path, index=False)
        print(f"\n已保存样本列表: {info_path}")
        print(f"  样本数: {len(info_df)}")
    
    # 计算全局类别统计（用于类别权重）
    print("\n计算类别分布统计...")
    global_class_stats = np.zeros(config["num_classes"], dtype=np.int64)
    
    for r in successful_results:
        if "class_stats" in r:
            global_class_stats += r["class_stats"]
    
    # 保存类别统计
    stats_df = pd.DataFrame({
        "class_id": range(config["num_classes"]),
        "class_name": config["class_names"],
        "total_voxels": global_class_stats,
        "percentage": global_class_stats / global_class_stats.sum() * 100
    })
    
    stats_path = output_root / "class_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n已保存类别统计: {stats_path}")
    print("\n类别分布:")
    print(stats_df.to_string(index=False))
    
    # 计算推荐的类别权重（inverse frequency）
    print("\n推荐的类别权重（用于训练配置）:")
    total_voxels = global_class_stats.sum()
    class_weights = []
    
    for i, (name, count) in enumerate(zip(config["class_names"], global_class_stats)):
        if count > 0:
            # 反频率权重，并归一化
            weight = total_voxels / (config["num_classes"] * count)
            class_weights.append(weight)
        else:
            class_weights.append(0.0)
    
    # 归一化权重，使背景权重为1
    if class_weights[0] > 0:
        normalized_weights = [w / class_weights[0] for w in class_weights]
    else:
        normalized_weights = class_weights
    
    print("\nclass_weights = [")
    for i, (name, w) in enumerate(zip(config["class_names"], normalized_weights)):
        print(f"    {w:.4f},  # {i}: {name}")
    print("]")
    
    # 保存权重到文件
    weights_dict = {
        "class_weights": normalized_weights,
        "class_names": config["class_names"]
    }
    import json
    weights_path = output_root / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"\n已保存类别权重: {weights_path}")
    
    print("\n" + "=" * 80)
    print("数据预处理完成！")
    print("=" * 80)
    print(f"\n下一步:")
    print(f"  1. 检查输出目录: {output_root}")
    print(f"  2. 更新训练配置文件中的 data_path 为: {output_root}")
    print(f"  3. 将上面的 class_weights 复制到 config_near.py")
    print(f"  4. 运行训练脚本: python near_repair.py")


# ============ 命令行接口 ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeAR 数据预处理脚本")
    parser.add_argument("--image_dir", type=str, help="ImageCAS 原始图像目录")
    parser.add_argument("--seg_dir", type=str, help="分割掩膜目录")
    parser.add_argument("--output_root", type=str, help="输出根目录")
    parser.add_argument("--target_resolution", type=str, help="目标分辨率，如 '1.0,1.0,1.0'（可选）")
    parser.add_argument("--n_workers", type=int, help="并行进程数")
    parser.add_argument("--hu_min", type=int, help="HU 值下限")
    parser.add_argument("--hu_max", type=int, help="HU 值上限")
    
    args = parser.parse_args()
    
    # 使用默认配置
    config = DEFAULT_CONFIG.copy()
    
    # 命令行参数覆盖
    if args.image_dir:
        config["image_dir"] = args.image_dir
    if args.seg_dir:
        config["seg_dir"] = args.seg_dir
    if args.output_root:
        config["output_root"] = args.output_root
    if args.target_resolution:
        config["target_resolution"] = tuple(map(float, args.target_resolution.split(",")))
    if args.n_workers:
        config["n_workers"] = args.n_workers
    if args.hu_min:
        config["hu_min"] = args.hu_min
    if args.hu_max:
        config["hu_max"] = args.hu_max
    
    # 运行
    main(config)
