#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocessing script to convert NIfTI cardiac CT data to NeAR format."""
import os
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from skimage.transform import resize as sk_resize
import warnings
warnings.filterwarnings('ignore')

DEFAULT_CONFIG = {
    "image_dir": "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/bboxed/images",
    "seg_dir": "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/bboxed/segmentations",
    "output_root": "/home/user/persistent/NeAR_fix_Public-Cardiac-CT-Dataset/dataset/near_format_data",
    "target_resolution": (128, 128, 128),
    "hu_min": -200,
    "hu_max": 500,
    "normalize_method": "minmax",
    "num_classes": 11,
    "class_names": [
        "Background", "Myocardium", "LA", "LV", "RA", "RV",
        "Aorta", "PA", "LAA", "Coronary", "PV"
    ],
    "n_workers": min(8, cpu_count() // 2),
}

def load_nifti(path):
    """Load NIfTI medical image."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return arr, spacing, img

def resample_volume(volume, original_spacing, target_spacing, is_label=False):
    """Resample 3D volume to target resolution."""
    if all(s >= 10 for s in target_spacing):
        target_shape = tuple(int(s) for s in target_spacing)
        if is_label:
            resampled = sk_resize(volume, target_shape, order=0,
                                  preserve_range=True, anti_aliasing=False).astype(volume.dtype)
        else:
            resampled = sk_resize(volume, target_shape, order=3,
                                  preserve_range=True).astype(np.float32)
        return resampled
    else:
        img = sitk.GetImageFromArray(volume)
        img.SetSpacing(original_spacing)
        original_size = img.GetSize()
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
        ]
        interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetDefaultPixelValue(0)
        resampled = resampler.Execute(img)
        return sitk.GetArrayFromImage(resampled)

def normalize_ct(ct_array, hu_min=-200, hu_max=500, method="minmax"):
    """
    Clip + normalize CT intensities to [0,1].
    """
    ct = np.clip(ct_array, hu_min, hu_max).astype(np.float32)
    if method == "minmax":
        ct = (ct - hu_min) / float(hu_max - hu_min + 1e-6)
    return ct

def validate_segmentation(seg_array, num_classes=11):
    """Validate segmentation label range."""
    unique_values = np.unique(seg_array)
    if np.max(unique_values) >= num_classes or np.min(unique_values) < 0:
        return False, unique_values
    return True, unique_values

def compute_class_statistics(seg_array, num_classes=11):
    """Compute voxel count per class."""
    stats = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        stats[c] = np.sum(seg_array == c)
    return stats

def process_single_case(args):
    """Process a single case (for multiprocessing)."""
    case_id, config = args
    try:
        img_path = Path(config["image_dir"]) / f"{case_id}.nii.img.nii.gz"
        seg_path = Path(config["seg_dir"]) / f"{case_id}.nii.img.nii.gz"
        if not img_path.exists():
            return {"case_id": case_id, "status": "missing_image", "error": str(img_path)}
        if not seg_path.exists():
            return {"case_id": case_id, "status": "missing_seg", "error": str(seg_path)}

        ct_array, ct_spacing, _ = load_nifti(img_path)
        seg_array, seg_spacing, _ = load_nifti(seg_path)

        is_valid, unique_vals = validate_segmentation(seg_array, config["num_classes"])
        if not is_valid:
            return {"case_id": case_id, "status": "invalid_seg", "error": f"Invalid labels: {unique_vals}"}

        if config["target_resolution"] is not None:
            target_spacing = config["target_resolution"]
            ct_array = resample_volume(ct_array, ct_spacing, target_spacing, is_label=False)
            seg_array = resample_volume(seg_array, seg_spacing, target_spacing, is_label=True)

        ct_norm = normalize_ct(ct_array, config["hu_min"], config["hu_max"], config["normalize_method"])
        seg_int = np.rint(seg_array).astype(np.int16)

        class_stats = compute_class_statistics(seg_int, config["num_classes"])
        app_out_path = Path(config["output_root"]) / "appearance" / f"{case_id}.npy"
        seg_out_path = Path(config["output_root"]) / "shape" / f"{case_id}.npy"
        app_out_path.parent.mkdir(parents=True, exist_ok=True)
        seg_out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(app_out_path, ct_norm)
        np.save(seg_out_path, seg_int)

        return {
            "case_id": case_id,
            "status": "success",
            "original_shape": ct_array.shape,
            "original_spacing": ct_spacing,
            "class_stats": class_stats
        }
    except Exception as e:
        return {"case_id": case_id, "status": "error", "error": str(e)}

def get_case_ids(image_dir):
    """Extract case IDs from image directory."""
    image_dir = Path(image_dir)
    case_ids = [f.name.replace(".nii.img.nii.gz", "") for f in sorted(image_dir.glob("*.nii.img.nii.gz"))]
    return case_ids

def main(config):
    """Main preprocessing function."""
    print("=" * 80)
    print("NeAR Data Preprocessing")
    print("=" * 80)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "appearance").mkdir(exist_ok=True)
    (output_root / "shape").mkdir(exist_ok=True)
    case_ids = get_case_ids(config["image_dir"])
    print(f"\nFound {len(case_ids)} samples")
    print(f"Output directory: {output_root}")
    print(f"Using {config['n_workers']} processes\n")
    process_args = [(cid, config) for cid in case_ids]
    print("Processing...")
    results = []
    if config["n_workers"] > 1:
        with Pool(config["n_workers"]) as pool:
            for result in tqdm(pool.imap(process_single_case, process_args), total=len(case_ids)):
                results.append(result)
    else:
        for args in tqdm(process_args):
            results.append(process_single_case(args))
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_results = [r for r in results if r["status"] != "success"]
    print(f"\nProcessing complete:\n  Success: {success_count}/{len(case_ids)}\n  Failed: {len(failed_results)}/{len(case_ids)}")
    if failed_results:
        print("\nFailed samples:")
        for r in failed_results[:10]:
            print(f"  - {r['case_id']}: {r['status']} ({r.get('error', '')})")

    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        info_df = pd.DataFrame([{"id": r["case_id"]} for r in successful_results])
        info_df.to_csv(output_root / "info.csv", index=False)
        print(f"\nSaved sample list: {output_root / 'info.csv'}")

    # class statistics
    global_class_stats = np.zeros(config["num_classes"], dtype=np.int64)
    for r in successful_results:
        if "class_stats" in r:
            global_class_stats += r["class_stats"]
    stats_df = pd.DataFrame({
        "class_id": range(config["num_classes"]),
        "class_name": config["class_names"],
        "total_voxels": global_class_stats,
        "percentage": global_class_stats / global_class_stats.sum() * 100
    })
    stats_path = output_root / "class_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved class statistics: {stats_path}")
    print("\nClass distribution:")
    print(stats_df.to_string(index=False))
    # class weights
    total_voxels = global_class_stats.sum()
    class_weights = [total_voxels / (config["num_classes"] * c) if c > 0 else 0.0 for c in global_class_stats]
    normalized_weights = [w / class_weights[0] if class_weights[0] > 0 else w for w in class_weights]
    print("\nclass_weights = [")
    for i, (name, w) in enumerate(zip(config["class_names"], normalized_weights)):
        print(f"    {w:.4f},  # {i}: {name}")
    print("]")
    import json
    weights_dict = {"class_weights": normalized_weights, "class_names": config["class_names"]}
    with open(output_root / "class_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"\nSaved class weights: {output_root / 'class_weights.json'}")
    print("\n" + "=" * 80)
    print("Data preprocessing complete!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeAR data preprocessing script")
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--seg_dir", type=str)
    parser.add_argument("--output_root", type=str)
    parser.add_argument("--target_resolution", type=str)
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--hu_min", type=int)
    parser.add_argument("--hu_max", type=int)
    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()
    if args.image_dir: config["image_dir"] = args.image_dir
    if args.seg_dir: config["seg_dir"] = args.seg_dir
    if args.output_root: config["output_root"] = args.output_root
    if args.target_resolution: config["target_resolution"] = tuple(map(float, args.target_resolution.split(",")))
    if args.n_workers: config["n_workers"] = args.n_workers
    if args.hu_min: config["hu_min"] = args.hu_min
    if args.hu_max: config["hu_max"] = args.hu_max
    main(config)
