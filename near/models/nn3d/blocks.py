import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Module):
    """3D Conv + Norm + Activation."""
    def __init__(self, in_ch, out_ch, norm=lambda c: nn.GroupNorm(8, c), activation=nn.LeakyReLU):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm = norm(out_ch) if norm is not None else nn.Identity()
        self.act = activation(inplace=True) if activation is nn.LeakyReLU else activation()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class LatentCodeUpsample(nn.Module):
    """Upsample latent code volume (trilinear) and reduce channels."""
    def __init__(self, in_ch, upsample_factor=2, channel_reduction=2,
                 norm=lambda c: nn.GroupNorm(8, c), activation=nn.LeakyReLU):
        super().__init__()
        out_ch = in_ch // channel_reduction
        self.up = nn.Upsample(scale_factor=upsample_factor, mode='trilinear', align_corners=True)
        self.proj = ConvNormAct(in_ch, out_ch, norm=norm, activation=activation)
    def forward(self, x):
        return self.proj(self.up(x))