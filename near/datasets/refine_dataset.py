# near/datasets/refine_dataset.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from scipy.ndimage import rotate, gaussian_filter, map_coordinates
from typing import Optional, Tuple

class CardiacMultiClassDataset(Dataset):
    """Cardiac multi-class segmentation dataset.
    
    Returns (index, appearance[1,D,H,W], seg[1,D,H,W]):
    - appearance: float32, HU clipped and normalized to [0,1] during preprocessing
    - seg: int16/long, values 0..10 (11 classes including background)
    
    Augmentations:
        - Random flip (3 axes)
        - Random rotation (±15°)
        - Elastic deformation
        - Intensity jitter
        - Gaussian noise
        - Gamma correction
    """

    def __init__(
        self,
        root: str,
        info_csv: Optional[str] = None,
        appearance_path: str = "appearance",
        shape_path: str = "shape",
        resolution: int = 128,
        n_samples: Optional[int] = None,
        normalize: bool = False,
        augment: bool = False,
        # Augmentation probabilities
        flip_prob: float = 0.5,
        rotate_prob: float = 0.3,
        elastic_prob: float = 0.2,
        intensity_prob: float = 0.5,
        noise_prob: float = 0.3,
        gamma_prob: float = 0.3,
        # Augmentation parameters
        rotate_angle_range: Tuple[float, float] = (-15, 15),
        elastic_alpha: float = 5.0,
        elastic_sigma: float = 3.0,
        intensity_scale: Tuple[float, float] = (0.9, 1.1),
        intensity_shift: Tuple[float, float] = (-0.1, 0.1),
        noise_std: float = 0.05,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
    ):
        """
        Args:
            root: near_format_data directory
            info_csv: Optional, specify train/val list (contains id or ROI_id column)
            appearance_path: Appearance volume directory name
            shape_path: Label volume directory name
            resolution: Target voxel size (isotropic)
            n_samples: Use only first n samples (for debugging)
            normalize: Apply [0,1] normalization again (usually False, as preprocessing is done)
            augment: Enable data augmentation
            
            # Augmentation probabilities
            flip_prob: Flip probability
            rotate_prob: Rotation probability
            elastic_prob: Elastic deformation probability
            intensity_prob: Intensity jitter probability
            noise_prob: Gaussian noise probability
            gamma_prob: Gamma correction probability
            
            # Augmentation parameters
            rotate_angle_range: Rotation angle range (degrees)
            elastic_alpha: Elastic deformation strength
            elastic_sigma: Elastic deformation smoothness
            intensity_scale: Intensity scaling range
            intensity_shift: Intensity shift range
            noise_std: Gaussian noise standard deviation
            gamma_range: Gamma range
        """
        self.root = root
        self.appearance_dir = os.path.join(root, appearance_path)
        self.shape_dir = os.path.join(root, shape_path)
        self.resolution = resolution
        self.normalize = normalize
        self.augment = augment
        
        # Augmentation settings
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.elastic_prob = elastic_prob
        self.intensity_prob = intensity_prob
        self.noise_prob = noise_prob
        self.gamma_prob = gamma_prob
        
        self.rotate_angle_range = rotate_angle_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.intensity_scale = intensity_scale
        self.intensity_shift = intensity_shift
        self.noise_std = noise_std
        self.gamma_range = gamma_range

        if info_csv is None:
            info_path = os.path.join(root, "info.csv")
        else:
            info_path = info_csv

        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Missing {info_path}")
        df = pd.read_csv(info_path)

        if "ROI_id" in df.columns:
            self.info = df[["ROI_id"]].copy()
            self.id_key = "ROI_id"
        elif "id" in df.columns:
            self.info = df[["id"]].copy()
            self.id_key = "id"
        else:
            raise ValueError("info csv must contain 'id' or 'ROI_id' column")

        if n_samples is not None:
            self.info = self.info.iloc[:n_samples].reset_index(drop=True)

    def __len__(self):
        return len(self.info)

    @staticmethod
    def _maybe_resize(vol: np.ndarray, tgt_size: Optional[tuple], order: int) -> np.ndarray:
        """Resize volume to target size."""
        if tgt_size is None:
            return vol
        if vol.shape == tgt_size:
            return vol
        return resize(
            vol, tgt_size, order=order, preserve_range=True, anti_aliasing=False
        ).astype(vol.dtype)

    def _random_flip(self, vol: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random flip along each axis."""
        if np.random.rand() < self.flip_prob:
            # Flip along depth axis
            if np.random.rand() < 0.5:
                vol = np.flip(vol, axis=0).copy()
                seg = np.flip(seg, axis=0).copy()
            # Flip along height axis
            if np.random.rand() < 0.5:
                vol = np.flip(vol, axis=1).copy()
                seg = np.flip(seg, axis=1).copy()
            # Flip along width axis
            if np.random.rand() < 0.5:
                vol = np.flip(vol, axis=2).copy()
                seg = np.flip(seg, axis=2).copy()
        return vol, seg

    def _random_rotate(self, vol: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random rotation in axial plane (Z axis)."""
        if np.random.rand() < self.rotate_prob:
            angle = np.random.uniform(*self.rotate_angle_range)
            # Rotate around Z-axis (affects X-Y plane)
            vol = rotate(vol, angle, axes=(1, 2), reshape=False, order=3, mode='nearest')
            seg = rotate(seg, angle, axes=(1, 2), reshape=False, order=0, mode='nearest')
            # Ensure segmentation remains integer
            seg = np.rint(seg).astype(seg.dtype)
        return vol, seg

    def _elastic_deformation(self, vol: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Elastic deformation following Ronneberger et al. (U-Net paper).
        
        Generate random displacement fields and apply to both image and label.
        """
        if np.random.rand() < self.elastic_prob:
            shape = vol.shape
            
            # Generate random displacement fields
            dx = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.elastic_sigma,
                mode="constant",
                cval=0
            ) * self.elastic_alpha
            
            dy = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.elastic_sigma,
                mode="constant",
                cval=0
            ) * self.elastic_alpha
            
            dz = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1),
                self.elastic_sigma,
                mode="constant",
                cval=0
            ) * self.elastic_alpha
            
            # Create meshgrid
            z, y, x = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij'
            )
            
            # Apply displacement
            indices = np.reshape(z + dz, (-1, 1)), \
                     np.reshape(y + dy, (-1, 1)), \
                     np.reshape(x + dx, (-1, 1))
            
            # Interpolate
            vol = map_coordinates(vol, indices, order=3, mode='nearest').reshape(shape)
            seg = map_coordinates(seg, indices, order=0, mode='nearest').reshape(shape)
            
            # Ensure segmentation remains integer
            seg = np.rint(seg).astype(seg.dtype)
        
        return vol, seg

    def _intensity_jitter(self, vol: np.ndarray) -> np.ndarray:
        """Random intensity scaling and shifting."""
        if np.random.rand() < self.intensity_prob:
            # Random scale
            scale = np.random.uniform(*self.intensity_scale)
            vol = vol * scale
            
            # Random shift
            shift = np.random.uniform(*self.intensity_shift)
            vol = vol + shift
            
            # Clip to [0, 1]
            vol = np.clip(vol, 0.0, 1.0)
        
        return vol

    def _add_gaussian_noise(self, vol: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        if np.random.rand() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, vol.shape)
            vol = vol + noise
            vol = np.clip(vol, 0.0, 1.0)
        return vol

    def _gamma_correction(self, vol: np.ndarray) -> np.ndarray:
        """Random gamma correction."""
        if np.random.rand() < self.gamma_prob:
            gamma = np.random.uniform(*self.gamma_range)
            # Apply gamma: I_out = I_in ^ gamma
            # Ensure all values are non-negative before power operation
            vol = np.clip(vol, 0.0, 1.0)
            vol = np.power(vol, gamma)
            vol = np.clip(vol, 0.0, 1.0)
        return vol

    def __getitem__(self, index):
        rid = self.info.loc[index, self.id_key]
        app_path = os.path.join(self.appearance_dir, f"{rid}.npy")
        seg_path = os.path.join(self.shape_dir, f"{rid}.npy")

        if not os.path.exists(app_path) or not os.path.exists(seg_path):
            raise FileNotFoundError(f"Missing {app_path} or {seg_path}")

        app = np.load(app_path).astype(np.float32)
        seg = np.load(seg_path).astype(np.int16)  # [D,H,W], values 0..10

        if self.normalize:
            vmin, vmax = app.min(), app.max()
            if vmax > vmin:
                app = (app - vmin) / (vmax - vmin + 1e-6)

        # Resize to target resolution
        tgt = (self.resolution, self.resolution, self.resolution)
        app = self._maybe_resize(app, tgt, order=3)  # cubic interpolation
        seg = self._maybe_resize(seg, tgt, order=0)  # nearest neighbor
        
        # Ensure segmentation is integer after resize
        seg = np.rint(seg).astype(np.int16)

        # Apply augmentations (order matters!)
        if self.augment:
            # 1. Geometric transforms (applied to both volume and segmentation)
            app, seg = self._random_flip(app, seg)
            app, seg = self._random_rotate(app, seg)
            app, seg = self._elastic_deformation(app, seg)
            
            # 2. Intensity transforms (only applied to appearance volume)
            app = self._intensity_jitter(app)
            app = self._add_gaussian_noise(app)
            app = self._gamma_correction(app)

        # Ensure contiguous and correct dtype
        app = np.ascontiguousarray(app).astype(np.float32)
        seg = np.ascontiguousarray(seg).astype(np.int16)

        # Convert to PyTorch tensors
        app_t = torch.from_numpy(app.copy()).unsqueeze(0).float()  # [1,D,H,W]
        seg_t = torch.from_numpy(seg.copy()).unsqueeze(0).long()   # [1,D,H,W]
        
        return index, app_t, seg_t

# Backward compatibility alias
RefineDataset = CardiacMultiClassDataset