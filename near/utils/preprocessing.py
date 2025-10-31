"""Image preprocessing utilities for medical images."""
import numpy as np
from scipy.ndimage import zoom

def pad_img(img, target_size, mode='constant', constant=0):
    """Pad image to target size.
    
    Args:
        img: Input image array
        target_size: Target dimensions (D, H, W)
        mode: Padding mode ('constant', 'edge', 'reflect', etc.)
        constant: Constant value for 'constant' mode
    
    Returns:
        Padded image
    """
    assert img.ndim == len(target_size)
    
    # Calculate padding for each dimension
    pad_width = []
    for i in range(img.ndim):
        diff = target_size[i] - img.shape[i]
        if diff < 0:
            raise ValueError(f"Image size {img.shape[i]} exceeds target {target_size[i]} in dim {i}")
        
        # Distribute padding evenly on both sides
        pad_before = diff // 2
        pad_after = diff - pad_before
        pad_width.append((pad_before, pad_after))
    
    # Apply padding
    if mode == 'constant':
        padded = np.pad(img, pad_width, mode=mode, constant_values=constant)
    else:
        padded = np.pad(img, pad_width, mode=mode)
    
    return padded

def crop_img(img, target_size):
    """Crop image to target size from center.
    
    Args:
        img: Input image array
        target_size: Target dimensions (D, H, W)
    
    Returns:
        Cropped image
    """
    assert img.ndim == len(target_size)
    
    # Calculate crop start indices (center crop)
    starts = []
    for i in range(img.ndim):
        diff = img.shape[i] - target_size[i]
        if diff < 0:
            raise ValueError(f"Image size {img.shape[i]} smaller than target {target_size[i]} in dim {i}")
        
        start = diff // 2
        starts.append(start)
    
    # Crop
    slices = tuple(slice(s, s + t) for s, t in zip(starts, target_size))
    cropped = img[slices]
    
    return cropped

def resize_img(img, target_size, order=1):
    """Resize image to target size using interpolation.
    
    Args:
        img: Input image array
        target_size: Target dimensions (D, H, W)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        Resized image
    """
    # Calculate zoom factors
    zoom_factors = [t / s for t, s in zip(target_size, img.shape)]
    
    # Apply zoom
    resized = zoom(img, zoom_factors, order=order, mode='nearest')
    
    return resized

def flip_img(img, axis):
    """Flip image along specified axis.
    
    Args:
        img: Input image array
        axis: Axis to flip (0, 1, or 2 for 3D)
    
    Returns:
        Flipped image
    """
    return np.flip(img, axis=axis)

def normalize_intensity(img, clip_range=None, percentile_range=None):
    """Normalize image intensities to [0, 1].
    
    Args:
        img: Input image array
        clip_range: Tuple (min, max) for clipping before normalization
        percentile_range: Tuple (low_pct, high_pct) for percentile-based clipping
    
    Returns:
        Normalized image in range [0, 1]
    """
    img = img.astype(np.float32)
    
    # Percentile-based clipping
    if percentile_range is not None:
        low_pct, high_pct = percentile_range
        vmin = np.percentile(img, low_pct)
        vmax = np.percentile(img, high_pct)
        img = np.clip(img, vmin, vmax)
    
    # Hard value clipping
    elif clip_range is not None:
        vmin, vmax = clip_range
        img = np.clip(img, vmin, vmax)
    
    # Normalize to [0, 1]
    img_min = img.min()
    img_max = img.max()
    
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    
    return img
