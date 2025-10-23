#!/usr/bin/env python3
"""
随机抽取修复后的分割图像，计算各类别体素占比
"""
import numpy as np
import nibabel as nib
from pathlib import Path
import random

# 类别名称
class_names = {
    0: 'Background',
    1: 'Myocardium',
    2: 'LA',
    3: 'LV',
    4: 'RA',
    5: 'RV',
    6: 'Aorta',
    7: 'PA',
    8: 'LAA',
    9: 'Coronary',
    10: 'PV'
}

# 设置随机种子以便复现
random.seed(42)
np.random.seed(42)

# 修复后的分割目录
seg_dir = Path("inference_results_fixed_weights/repaired_segmentations")

# 获取所有文件
all_files = sorted(list(seg_dir.glob("*.nii.gz")))
print(f"总共找到 {len(all_files)} 个文件")

# 随机抽取10个
if len(all_files) < 10:
    selected_files = all_files
    print(f"文件数不足10个，使用所有 {len(all_files)} 个文件")
else:
    selected_files = random.sample(all_files, 10)
    print(f"随机抽取 10 个文件")

print("\n" + "=" * 120)
print("修复后分割图像 - 各类别体素占比分析")
print("=" * 120)

# 存储所有结果
all_results = []

for idx, file_path in enumerate(selected_files, 1):
    print(f"\n[{idx}/10] 分析: {file_path.name}")
    print("-" * 120)
    
    # 读取分割图像
    img = nib.load(file_path)
    data = img.get_fdata().astype(np.int32)
    
    # 图像尺寸
    shape = data.shape
    total_voxels = np.prod(shape)
    
    print(f"图像尺寸: {shape[0]} × {shape[1]} × {shape[2]} = {total_voxels:,} voxels")
    print()
    
    # 统计每个类别的体素数
    class_counts = {}
    for class_id in range(11):
        count = np.sum(data == class_id)
        percentage = count / total_voxels * 100
        class_counts[class_id] = {
            'count': count,
            'percentage': percentage
        }
    
    # 打印表格
    print(f"{'类别':<15} {'体素数':>15} {'占比':>10} {'可视化'}")
    print("-" * 120)
    
    result_dict = {'file': file_path.name}
    
    for class_id in range(11):
        count = class_counts[class_id]['count']
        pct = class_counts[class_id]['percentage']
        
        # 创建可视化条形图
        bar_length = int(pct / 2)  # 每2%一个字符
        bar = '█' * bar_length
        
        # 标记异常值
        marker = ""
        if class_id == 0 and (pct < 70 or pct > 90):
            marker = " ⚠️ 背景异常"
        elif class_id == 9 and pct > 5:
            marker = " ⚠️ Class 9过高"
        elif pct > 20 and class_id not in [0]:
            marker = " ⚠️ 占比过高"
        
        print(f"{class_names[class_id]:<15} {count:>15,} {pct:>9.2f}% {bar}{marker}")
        
        result_dict[f'class_{class_id}'] = pct
    
    all_results.append(result_dict)
    
    # 验证总和
    total_check = sum(class_counts[i]['percentage'] for i in range(11))
    print("-" * 120)
    print(f"总和验证: {total_check:.2f}% (应该 = 100.00%)")

# 汇总统计
print("\n" + "=" * 120)
print("汇总统计 (10个样本的平均占比)")
print("=" * 120)

avg_percentages = {}
for class_id in range(11):
    values = [r[f'class_{class_id}'] for r in all_results]
    avg = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    avg_percentages[class_id] = {
        'mean': avg,
        'std': std,
        'min': min_val,
        'max': max_val
    }

print(f"{'类别':<15} {'平均占比':>12} {'标准差':>10} {'最小':>10} {'最大':>10} {'评价'}")
print("-" * 120)

for class_id in range(11):
    mean = avg_percentages[class_id]['mean']
    std = avg_percentages[class_id]['std']
    min_val = avg_percentages[class_id]['min']
    max_val = avg_percentages[class_id]['max']
    
    # 评价
    status = ""
    if class_id == 0:
        if 70 <= mean <= 90:
            status = "✅ 正常"
        else:
            status = "⚠️ 异常"
    elif class_id == 9:
        if mean < 5:
            status = "✅ 正常"
        else:
            status = "⚠️ 过高"
    else:
        if mean < 20:
            status = "✅ 正常"
        else:
            status = "⚠️ 过高"
    
    print(f"{class_names[class_id]:<15} {mean:>11.2f}% {std:>9.2f}% {min_val:>9.2f}% {max_val:>9.2f}% {status}")

print("=" * 120)

# 关键指标检查
print("\n🎯 关键指标检查:")
bg_mean = avg_percentages[0]['mean']
c9_mean = avg_percentages[9]['mean']

print(f"✅ 背景占比: {bg_mean:.2f}% (正常范围: 70-90%)")
if 70 <= bg_mean <= 90:
    print("   ✅ 背景预测正常！")
else:
    print(f"   ⚠️ 背景占比异常！(实际: {bg_mean:.2f}%)")

print(f"✅ Class 9 (Coronary) 占比: {c9_mean:.2f}% (正常范围: <5%)")
if c9_mean < 5:
    print("   ✅ Class 9 预测正常！")
else:
    print(f"   ⚠️ Class 9 占比过高！(实际: {c9_mean:.2f}%)")

print("\n🎉 结论:")
if 70 <= bg_mean <= 90 and c9_mean < 5:
    print("✅ 修复后的模型预测结果正常！")
    print("✅ 背景和 Class 9 的占比都在合理范围内")
    print("✅ 权重修复成功！")
else:
    print("⚠️ 仍存在问题，需要进一步检查")

print("=" * 120)
