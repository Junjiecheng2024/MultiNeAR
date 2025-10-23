import torch
import torch.nn.functional as F

def generate_meshgrid_3d(res):
    """Uniform grid in [-1,1]^3 as (D,H,W,3)."""
    z = torch.linspace(-1, 1, res)
    zz, yy, xx = torch.meshgrid(z, z, z, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1)  # (D,H,W,3)
    return grid

class GatherGridsFromVolumes:
    """
    Create grids (with optional noise) and sample volumes:
    - appearance: trilinear
    - labels: nearest (call-site chooses mode)
    """
    def __init__(self, resolution, grid_noise=None, uniform_grid_noise=True):
        self.resolution = resolution
        self.grid_noise = grid_noise
        self.uniform_grid_noise = uniform_grid_noise

    def __call__(self, volume):
        B = volume.shape[0]
        base = generate_meshgrid_3d(self.resolution).to(volume.device)  # (D,H,W,3)
        base = base.unsqueeze(0).repeat(B, 1, 1, 1, 1)                  # [B,D,H,W,3]
        if self.grid_noise is not None and self.grid_noise > 0:
            noise = torch.empty_like(base).uniform_(-self.grid_noise, self.grid_noise) \
                    if self.uniform_grid_noise else torch.randn_like(base) * self.grid_noise
            grid = torch.clamp(base + noise, -1, 1)
        else:
            grid = base
        # sample (caller decides mode for labels)
        sampled = F.grid_sample(volume, grid, mode='bilinear', align_corners=True)
        return base, grid, sampled