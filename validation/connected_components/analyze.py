"""Connected components analysis for cardiac segmentation validation."""
import os
import csv
import numpy as np
import nibabel as nib
import cc3d

def load_label(path):
    """Load label volume from NIfTI file."""
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    return data.astype(np.int16)

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
    
    labeled = cc3d.connected_components(binary_mask.astype(np.uint8), connectivity=connectivity)
    
    component_sizes = np.bincount(labeled.ravel())
    component_sizes = component_sizes[1:]
    
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

def run_dataset(before_dir, after_dir, class_ids, out_csv_path):
    """Run connected components analysis on paired before/after cases.
    
    Args:
        before_dir: Directory containing baseline segmentations
        after_dir: Directory containing repaired segmentations
        class_ids: List of class IDs to analyze
        out_csv_path: Output CSV file path
    """
    import sys
    cases = sorted([f for f in os.listdir(before_dir) if f.endswith((".nii", ".nii.gz"))])
    total_cases = len(cases)
    print(f"Processing {total_cases} cases...", flush=True)
    
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    csvfile = open(out_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csvfile,
        fieldnames=["case","phase","class_id","n_components","total_voxels","largest_area","largest_ratio"]
    )
    writer.writeheader()
    csvfile.flush()
    
    try:
        for idx, fname in enumerate(cases, 1):
            if idx % 10 == 0 or idx == 1:
                print(f"  [{idx}/{total_cases}] {fname}", flush=True)
                sys.stdout.flush()
            
            before_path = os.path.join(before_dir, fname)
            
            case_id = fname
            if case_id.endswith('.nii.img.nii.gz'):
                case_id = case_id.replace('.nii.img.nii.gz', '')
            elif case_id.endswith('.nii.gz'):
                case_id = case_id.replace('.nii.gz', '')
            elif case_id.endswith('.nii'):
                case_id = case_id.replace('.nii', '')
            
            after_path = os.path.join(after_dir, f"{case_id}.nii.gz")
            if not os.path.exists(after_path):
                after_path = os.path.join(after_dir, f"{case_id}.nii.img.nii.gz")
                if not os.path.exists(after_path):
                    after_path = os.path.join(after_dir, f"{case_id}.nii")
                    if not os.path.exists(after_path):
                        print(f"[WARN] after file missing for case {case_id}, skip.", flush=True)
                        continue

            before_lab = load_label(before_path)
            after_lab  = load_label(after_path)

            before_stats = stats_per_case(before_lab, class_ids, connectivity=26)
            after_stats  = stats_per_case(after_lab,  class_ids, connectivity=26)

            for phase, per_class in [("before", before_stats), ("after", after_stats)]:
                for cid, st in per_class.items():
                    writer.writerow({
                        "case": case_id,
                        "phase": phase,
                        "class_id": cid,
                        "n_components": st["n_components"],
                        "total_voxels": st["total_voxels"],
                        "largest_area": st["largest_area"],
                        "largest_ratio": round(st["largest_ratio"], 6),
                    })
            csvfile.flush()
        
        print(f"\n[OK] Completed! Results saved to: {out_csv_path}", flush=True)
    finally:
        csvfile.close()

if __name__ == "__main__":
    before_dir = "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/matched_dataset/segmentations"
    after_dir  = "/home/user/persistent/3dUNet_val/labelsTr_repaired_resized"
    
    class_ids  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    out_csv    = "./data/cc_stats.csv"
    
    run_dataset(before_dir, after_dir, class_ids, out_csv)

