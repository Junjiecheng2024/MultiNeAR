#!/usr/bin/env python3
"""Resize repaired labels from 128^3 to original image dimensions."""
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
from tqdm import tqdm

def resize_label_to_original(repaired_path, original_path, output_path):
    """Resize repaired labels to match original image dimensions.
    
    Args:
        repaired_path: Path to repaired label (128^3)
        original_path: Path to original image (for target shape)
        output_path: Path for output
    
    Returns:
        target_shape: Original image shape
        current_shape: Repaired label shape
    """
    orig_img = nib.load(original_path)
    target_shape = orig_img.shape
    
    repaired_img = nib.load(repaired_path)
    repaired_data = repaired_img.get_fdata().astype(np.int32)
    current_shape = repaired_data.shape
    
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    
    resized_data = zoom(repaired_data, zoom_factors, order=0, mode='nearest')
    resized_data = resized_data[:target_shape[0], :target_shape[1], :target_shape[2]]
    
    resized_img = nib.Nifti1Image(
        resized_data.astype(np.int16),
        affine=orig_img.affine,
        header=orig_img.header
    )
    
    nib.save(resized_img, output_path)
    
    return target_shape, current_shape

def main():
    base_dir = Path("/home/user/persistent/3dUNet_val")
    images_dir = base_dir / "imagesTr"
    repaired_dir = base_dir / "labelsTr_repaired"
    output_dir = base_dir / "labelsTr_repaired_resized"
    
    image_files = sorted(list(images_dir.glob("*.nii.gz")))
    
    print("=" * 80)
    print("Resizing repaired labels to original image dimensions")
    print("=" * 80)
    print(f"Original images: {images_dir}")
    print(f"Repaired labels: {repaired_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files: {len(image_files)}")
    print("=" * 80)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    success_count = 0
    error_count = 0
    size_stats = {}
    
    for img_file in tqdm(image_files, desc="Resizing"):
        try:
            file_id = img_file.stem.split('.')[0]
            repaired_file = repaired_dir / f"{file_id}.nii.gz"
            
            if not repaired_file.exists():
                print(f"\nWarning: Repaired label not found: {repaired_file}")
                error_count += 1
                continue
            
            output_file = output_dir / f"{file_id}.nii.gz"
            
            target_shape, current_shape = resize_label_to_original(
                repaired_file, img_file, output_file
            )
            
            size_key = f"{target_shape[0]}x{target_shape[1]}x{target_shape[2]}"
            size_stats[size_key] = size_stats.get(size_key, 0) + 1
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print("Processing completed")
    print("=" * 80)
    print(f"Success: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Total: {len(image_files)}")
    
    print("\nSize distribution:")
    for size, count in sorted(size_stats.items()):
        print(f"  {size}: {count} files")
    
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
