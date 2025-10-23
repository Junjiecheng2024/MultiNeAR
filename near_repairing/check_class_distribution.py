#!/usr/bin/env python3
"""Randomly sample repaired segmentations and compute class distribution."""
import numpy as np
import nibabel as nib
from pathlib import Path
import random

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

random.seed(42)
np.random.seed(42)

seg_dir = Path("inference_results_fixed_weights/repaired_segmentations")

all_files = sorted(list(seg_dir.glob("*.nii.gz")))
print(f"Found {len(all_files)} files in total")

if len(all_files) < 10:
    selected_files = all_files
    print(f"Less than 10 files available, using all {len(all_files)} files")
else:
    selected_files = random.sample(all_files, 10)
    print(f"Randomly selected 10 files")

print("\n" + "=" * 120)
print("Repaired Segmentation - Class Distribution Analysis")
print("=" * 120)

all_results = []

for idx, file_path in enumerate(selected_files, 1):
    print(f"\n[{idx}/10] Analyzing: {file_path.name}")
    print("-" * 120)
    
    img = nib.load(file_path)
    data = img.get_fdata().astype(np.int32)
    
    shape = data.shape
    total_voxels = np.prod(shape)
    
    print(f"Image dimensions: {shape[0]} × {shape[1]} × {shape[2]} = {total_voxels:,} voxels")
    print()
    
    class_counts = {}
    for class_id in range(11):
        count = np.sum(data == class_id)
        percentage = count / total_voxels * 100
        class_counts[class_id] = {
            'count': count,
            'percentage': percentage
        }
    
    print(f"{'Class':<15} {'Voxel Count':>15} {'Percentage':>10} {'Visualization'}")
    print("-" * 120)
    
    result_dict = {'file': file_path.name}
    
    for class_id in range(11):
        count = class_counts[class_id]['count']
        pct = class_counts[class_id]['percentage']
        
        bar_length = int(pct / 2)
        bar = '█' * bar_length
        
        marker = ""
        if class_id == 0 and (pct < 70 or pct > 90):
            marker = " ⚠️ Abnormal background"
        elif class_id == 9 and pct > 5:
            marker = " ⚠️ Class 9 too high"
        elif pct > 20 and class_id not in [0]:
            marker = " ⚠️ Percentage too high"
        
        print(f"{class_names[class_id]:<15} {count:>15,} {pct:>9.2f}% {bar}{marker}")
        
        result_dict[f'class_{class_id}'] = pct
    
    all_results.append(result_dict)

print("\n" + "=" * 120)
print("Statistical Summary")
print("=" * 120)

import pandas as pd
results_df = pd.DataFrame(all_results)

for class_id in range(11):
    col_name = f'class_{class_id}'
    mean_pct = results_df[col_name].mean()
    std_pct = results_df[col_name].std()
    min_pct = results_df[col_name].min()
    max_pct = results_df[col_name].max()
    
    print(f"{class_names[class_id]:<15} Mean: {mean_pct:>6.2f}%  "
          f"Std: {std_pct:>6.2f}%  Range: [{min_pct:>6.2f}%, {max_pct:>6.2f}%]")

output_csv = "class_distribution_analysis.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nDetailed results saved to: {output_csv}")

print("\n" + "=" * 120)
print("Analysis Complete")
print("=" * 120)
