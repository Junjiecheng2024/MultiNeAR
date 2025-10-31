"""Connected components analysis for cardiac segmentation validation.

Purpose: Measure whether each anatomical structure (LV, LA, RV, etc.) remains 
connected after reconstruction. Ideally, each structure should be a single 
connected component (CC=1.00).

Metrics:
- Number of connected components per class
- Average CC across all classes (1-10, excluding background)
- Dataset-level statistics

Lower CC values indicate better reconstruction quality.
"""
import os
import csv
import pandas as pd
import numpy as np
import nibabel as nib
import cc3d
from collections import defaultdict

def load_label(path):
    """Load a NIfTI label file."""
    img = nib.load(path)
    return np.asanyarray(img.dataobj).astype(np.uint8)

def stats_one_class(binary_mask, connectivity=26):
    """Compute connected components statistics for one class.
    
    Args:
        binary_mask: Binary 3D numpy array (0/1)
        connectivity: Neighborhood connectivity (6, 18, or 26)
    
    Returns:
        Dictionary with n_components, total_voxels, largest_area, largest_ratio
    """
    if binary_mask.sum() == 0:
        return {
            "n_components": 0,
            "total_voxels": 0,
            "largest_area": 0,
            "largest_ratio": 0.0,
        }
    
    # Convert to uint8 for cc3d (more efficient)
    labeled = cc3d.connected_components(binary_mask.astype(np.uint8), connectivity=connectivity)
    
    # Fast bincount for component sizes
    component_sizes = np.bincount(labeled.ravel())
    component_sizes = component_sizes[1:]  # Remove background
    
    total = int(binary_mask.sum())
    n_comp = len(component_sizes)
    largest = int(component_sizes.max()) if n_comp > 0 else 0
    ratio = float(largest) / float(total) if total > 0 else 0.0
    
    return {
        "n_components": n_comp,
        "total_voxels": total,
        "largest_area": largest,
        "largest_ratio": ratio,
    }

def stats_per_case(label_vol, class_ids, connectivity=26):
    """Compute statistics for all classes in one case.
    
    Args:
        label_vol: Integer label volume
        class_ids: List of class IDs to analyze
        connectivity: Neighborhood connectivity (default 26)
    
    Returns:
        Dictionary mapping class_id to statistics
    """
    results = {}
    for cid in class_ids:
        binary = (label_vol == cid)
        results[cid] = stats_one_class(binary, connectivity=connectivity)
    return results

if __name__ == "__main__":
    # Paths for connectivity analysis on inference results
    # Use ORIGINAL data (not bboxed) as GT to include all 11 classes
    before_dir = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/original/segmentations"
    after_dir  = "/home/user/persistent/MultiNeAR/inference_results/v2/repaired_segmentations"
    info_csv   = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data/info.csv"
    
    # Analyze all 10 cardiac structures (excluding background class 0)
    class_ids  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Output CSV with connectivity statistics
    out_csv    = "/home/user/persistent/MultiNeAR/inference_results/v2/connectivity_analysis_full.csv"
    
    print("\n" + "="*80)
    print("Connected Components Analysis - NeAR Inference Results")
    print("="*80)
    print(f"Original labels:  {before_dir}")
    print(f"Repaired labels:  {after_dir}")
    print(f"ID mapping:       {info_csv}")
    print(f"Output CSV:       {out_csv}")
    print(f"Analyzing classes: {class_ids} (1=Myocardium to 10=PV)")
    print(f"Connectivity:     26-connected (3D)")
    print("="*80 + "\n")
    
    # Load ID mapping
    print("Loading ID mapping from info.csv...")
    info_df = pd.read_csv(info_csv)
    case_ids = info_df['id'].tolist()
    print(f"Loaded {len(case_ids)} case IDs\n")
    
    # Create output CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    csvfile = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csvfile,
        fieldnames=["case_idx", "original_id", "phase", "class_id", "n_components", "total_voxels", "largest_area", "largest_ratio"]
    )
    writer.writeheader()
    csvfile.flush()
    
    # Track statistics for summary
    phase_stats = {
        "before": defaultdict(list),
        "after": defaultdict(list)
    }
    
    successful_cases = 0
    
    import time
    start_time = time.time()
    
    try:
        print(f"Processing {len(case_ids)} cases...\n")
        for case_idx, original_id in enumerate(case_ids):
            case_start = time.time()
            
            if (case_idx + 1) % 50 == 0 or case_idx == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (case_idx + 1) if case_idx > 0 else 0
                eta = avg_time * (len(case_ids) - case_idx - 1)
                print(f"  [{case_idx + 1}/{len(case_ids)}] case_{case_idx:03d} (ID={original_id}) "
                      f"[Avg: {avg_time:.1f}s/case, ETA: {eta/60:.1f}min]", flush=True)
            
            # Original label path
            before_path = os.path.join(before_dir, f"{original_id}.nii.img.nii.gz")
            if not os.path.exists(before_path):
                print(f"[WARN] Original file missing: {original_id}", flush=True)
                continue
            
            # Repaired label path (now using original ID like inference.py)
            after_path = os.path.join(after_dir, f"{original_id}.nii.gz")
            if not os.path.exists(after_path):
                print(f"[WARN] Repaired file missing: {original_id}", flush=True)
                continue
            
            # Load labels
            load_start = time.time()
            before_lab = load_label(before_path)
            after_lab = load_label(after_path)
            load_time = time.time() - load_start
            
            # Compute statistics
            comp_start = time.time()
            before_stats = stats_per_case(before_lab, class_ids, connectivity=26)
            after_stats = stats_per_case(after_lab, class_ids, connectivity=26)
            comp_time = time.time() - comp_start
            
            # Debug: Print timing for first few cases
            if case_idx < 3:
                case_time = time.time() - case_start
                print(f"    Case {case_idx}: Load={load_time:.2f}s, Compute={comp_time:.2f}s, Total={case_time:.2f}s", flush=True)
            
            # Save to CSV
            for phase, per_class in [("before", before_stats), ("after", after_stats)]:
                for cid, st in per_class.items():
                    writer.writerow({
                        "case_idx": case_idx,
                        "original_id": original_id,
                        "phase": phase,
                        "class_id": cid,
                        "n_components": st["n_components"],
                        "total_voxels": st["total_voxels"],
                        "largest_area": st["largest_area"],
                        "largest_ratio": round(st["largest_ratio"], 6),
                    })
                    
                    # Accumulate for summary
                    if st["total_voxels"] > 0:
                        phase_stats[phase][cid].append(st["n_components"])
            
            csvfile.flush()
            successful_cases += 1
        
        print(f"\n{'='*80}")
        print(f"Successfully processed {successful_cases}/{len(case_ids)} cases")
        print(f"{'='*80}\n")
        
        print(f"{'='*80}")
        print("SUMMARY STATISTICS - Connected Components Analysis")
        print(f"{'='*80}\n")
        
        # Class names mapping
        class_names = {
            1: "Myocardium", 2: "LA", 3: "LV", 4: "RA", 5: "RV",
            6: "Aorta", 7: "PA", 8: "LAA", 9: "Coronary", 10: "PV"
        }
        
        print(f"{'Class':<12} {'Name':<15} {'Before CC':<12} {'After CC':<12} {'Change':<10}")
        print("-" * 80)
        
        overall_before = []
        overall_after = []
        
        for cid in class_ids:
            before_ccs = phase_stats["before"][cid]
            after_ccs = phase_stats["after"][cid]
            
            if len(before_ccs) > 0 and len(after_ccs) > 0:
                mean_before = np.mean(before_ccs)
                mean_after = np.mean(after_ccs)
                change = mean_after - mean_before
                
                overall_before.extend(before_ccs)
                overall_after.extend(after_ccs)
                
                change_str = f"{change:+.2f}" if abs(change) > 0.01 else "±0.00"
                print(f"Class {cid:<5} {class_names.get(cid, 'Unknown'):<15} "
                      f"{mean_before:<12.2f} {mean_after:<12.2f} {change_str:<10}")
        
        print("-" * 80)
        if overall_before and overall_after:
            avg_before = np.mean(overall_before)
            avg_after = np.mean(overall_after)
            change = avg_after - avg_before
            change_str = f"{change:+.2f}" if abs(change) > 0.01 else "±0.00"
            
            print(f"{'AVERAGE':<12} {'(All Classes)':<15} "
                  f"{avg_before:<12.2f} {avg_after:<12.2f} {change_str:<10}\n")
            
            print(f"Interpretation:")
            print(f"  - Ideal CC value: 1.00 (single connected component per structure)")
            print(f"  - Before (baseline): {avg_before:.2f} components per structure")
            print(f"  - After (repaired):  {avg_after:.2f} components per structure")
            
            if avg_after < avg_before:
                improvement = ((avg_before - avg_after) / avg_before) * 100
                print(f"  ✅ Improvement: {improvement:.1f}% reduction in fragmentation")
            elif avg_after > avg_before:
                degradation = ((avg_after - avg_before) / avg_before) * 100
                print(f"  ⚠️  Degradation: {degradation:.1f}% increase in fragmentation")
            else:
                print(f"  ↔️  No change in connectivity")
        
        print(f"\n{'='*80}")
        print(f"[OK] Completed! Results saved to: {out_csv}")
        print(f"{'='*80}\n")
    finally:
        csvfile.close()

