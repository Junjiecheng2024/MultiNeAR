import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

class CardiacMultiClassDataset(Dataset):
    """Return (index, appearance[1,D,H,W], seg[1,D,H,W])."""

    def __init__(self, root, appearance_path="appearance", shape_path="shape",
                 resolution=128, n_samples=None, normalize=True):
        self.root = root
        self.appearance_dir = os.path.join(root, appearance_path)
        self.shape_dir = os.path.join(root, shape_path)
        self.resolution = resolution
        self.normalize = normalize

        info_path = os.path.join(root, "info.csv")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Missing {info_path}")
        df = pd.read_csv(info_path)
        if "ROI_id" in df.columns:
            self.info = df[["ROI_id"]]
            self.id_key = "ROI_id"
        elif "id" in df.columns:
            self.info = df[["id"]]
            self.id_key = "id"
        else:
            raise ValueError("info.csv must have ROI_id or id")
        if n_samples is not None:
            self.info = self.info.iloc[:n_samples]

    def __len__(self):
        return len(self.info)

    def _maybe_resize(self, vol, order):
        if self.resolution is None:
            return vol
        tgt = (self.resolution,) * 3
        if vol.shape == tgt:
            return vol
        return resize(vol, tgt, order=order, preserve_range=True, anti_aliasing=False).astype(vol.dtype)

    def __getitem__(self, index):
        rid = self.info.loc[index, self.id_key]
        app_path = os.path.join(self.appearance_dir, f"{rid}.npy")
        seg_path = os.path.join(self.shape_dir, f"{rid}.npy")
        if not os.path.exists(app_path) or not os.path.exists(seg_path):
            raise FileNotFoundError(f"Missing {app_path} or {seg_path}")

        app = np.load(app_path).astype(np.float32)     # CT
        seg = np.load(seg_path).astype(np.int16)       # labels 0..10

        if self.normalize:
            vmin, vmax = app.min(), app.max()
            if vmax > vmin:
                app = (app - vmin) / (vmax - vmin)

        app = self._maybe_resize(app, order=3)         # cubic for appearance
        seg = self._maybe_resize(seg, order=0)         # nearest for labels

        app_t = torch.tensor(app, dtype=torch.float32).unsqueeze(0)  # [1,D,H,W]
        seg_t = torch.tensor(seg, dtype=torch.long).unsqueeze(0)     # [1,D,H,W]
        return index, app_t, seg_t