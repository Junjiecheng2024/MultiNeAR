# Connected Components Analysis

This directory contains scripts for analyzing the topological changes in cardiac segmentations before and after NeAR repair.

## Overview

Connected components analysis quantifies topological fragmentation in 3D segmentations. For each anatomical structure, we compute:
- Number of connected components
- Total voxel count
- Largest component size
- Largest component ratio (size / total)

This analysis demonstrates how NeAR repairs topological defects (e.g., merging disconnected fragments into single coherent structures).

## Files

- `analyze.py`: Compute connected components statistics for baseline vs. repaired segmentations

## Usage

Run connected components analysis:
```bash
python analyze.py
```

The script will:
1. Load paired baseline and repaired segmentations
2. Compute connected components for each anatomical class (1-10)
3. Export results to `data/cc_stats.csv`

## Output Format

CSV file with the following columns:
- `case`: Case ID
- `phase`: "before" (baseline) or "after" (repaired)
- `class_id`: Anatomical structure ID (1-10)
- `n_components`: Number of disconnected components
- `total_voxels`: Total structure volume in voxels
- `largest_area`: Size of largest component
- `largest_ratio`: Ratio of largest component to total volume

## Anatomical Classes

1. Left Ventricle (LV)
2. Right Ventricle (RV)
3. Left Atrium (LA)
4. Right Atrium (RA)
5. Myocardium
6. Aorta
7. Pulmonary Artery
8. Coronary Artery
9. Superior Vena Cava (SVC)
10. Inferior Vena Cava (IVC)

## Expected Results

After NeAR repair:
- **Reduced n_components**: Fewer disconnected fragments
- **Increased largest_ratio**: Dominant component represents most of the structure
- **Improved topology**: Structures become more coherent and anatomically plausible

## Technical Details

- **Algorithm**: CC3D library (fast C++ implementation)
- **Connectivity**: 26-neighborhood (face + edge + vertex connected)
- **Processing**: Streaming mode to handle large datasets efficiently
