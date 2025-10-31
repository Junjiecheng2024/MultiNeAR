# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from near.models.nn3d.blocks import LatentCodeUpsample, ConvNormAct

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def make_groupnorm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Robust GroupNorm: choose groups <= max_groups and dividing num_channels."""
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)

DEFAULT = {
    "norm": make_groupnorm,
    "activation": lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True),
}

def kaiming_init(module: nn.Module):
    """Kaiming init for Conv3d/Linear; zero bias."""
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# ---------------------------------------------------------------------
# (Optional) Fourier positional encoding for coords (disabled by default)
# ---------------------------------------------------------------------
class FourierFeatures3D(nn.Module):
    def __init__(self, num_bands: int = 6, include_input: bool = True):
        super().__init__()
        self.num_bands = num_bands
        self.include_input = include_input

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [B, 3, D, H, W] in [-1, 1]
        returns: [B, C_ff, D, H, W]
        """
        outs = []
        if self.include_input:
            outs.append(coords)
        # frequencies: 1,2,4,...,2^(num_bands-1)
        freqs = 2.0 ** torch.arange(self.num_bands, device=coords.device, dtype=coords.dtype)
        freqs = freqs.view(1, -1, 1, 1, 1)  # [1, B, 1, 1, 1]
        x = coords.unsqueeze(2)  # [B, 3, 1, D, H, W]
        # broadcast multiply on channel=3 then freq bands
        x = x * freqs  # [B, 3, B, D, H, W]
        outs.append(torch.sin(x).flatten(1, 2))
        outs.append(torch.cos(x).flatten(1, 2))
        return torch.cat(outs, dim=1)

# ---------------------------------------------------------------------
# Implicit decoder
# ---------------------------------------------------------------------
class ImplicitDecoder(nn.Module):
    """
    Implicit decoder: latent z → multi-scale 3D features → sample at grid → point-MLP → K logits.
    
    Pipeline:
    - Multi-scale volume features (4 scales)
    - Grid sampling at specified coordinates
    - Concatenate {coords (+Fourier optional), multi-scale sampled features, (optional) appearance}
    - 1x1x1 MLP produces K-class logits
    """
    def __init__(
        self,
        latent_dimension: int,
        out_channels: int,
        norm=DEFAULT["norm"],
        activation=DEFAULT["activation"],
        decoder_channels=(128, 96, 64, 32),
        appearance: bool = True,
        use_fourier: bool = False,
        fourier_bands: int = 6,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.appearance = appearance
        self.use_fourier = use_fourier

        # Latent z -> pyramid features
        self.decoder_1 = nn.Sequential(
            LatentCodeUpsample(latent_dimension, upsample_factor=2, channel_reduction=2, norm=None, activation=activation),
            LatentCodeUpsample(latent_dimension // 2, upsample_factor=2, channel_reduction=2, norm=norm, activation=activation),
            ConvNormAct(latent_dimension // 4, decoder_channels[0], norm=norm, activation=activation),
        )
        self.decoder_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[0], decoder_channels[1], norm=norm, activation=activation),
            ConvNormAct(decoder_channels[1], decoder_channels[1], norm=norm, activation=activation),
        )
        self.decoder_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[1], decoder_channels[2], norm=norm, activation=activation),
            ConvNormAct(decoder_channels[2], decoder_channels[2], norm=norm, activation=activation),
        )
        self.decoder_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[2], decoder_channels[3], norm=norm, activation=activation),
            ConvNormAct(decoder_channels[3], decoder_channels[3], norm=norm, activation=activation),
        )

        # coord embedding
        if self.use_fourier:
            self.ff = FourierFeatures3D(num_bands=fourier_bands, include_input=True)
            coord_ch = 3 + 3 * 2 * fourier_bands  # xyz + sin/cos bands
        else:
            self.ff = None
            coord_ch = 3

        # Concatenate channels: coords(+Fourier) + multi-scale + appearance(optional)
        in_ch = coord_ch + sum(decoder_channels) + (1 if appearance else 0)

        self.implicit_mlp = nn.Sequential(
            nn.Conv3d(in_ch, 128, 1), norm(128), activation(),  # Expand to 128
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(128, 128, 1), norm(128), activation(),
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(128, 64, 1),  norm(64), activation(),
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(64, out_channels, 1),
        )

        kaiming_init(self)

    def forward(self, latent_code, grid, appearance):
        """
        latent_code: [B, C]
        grid:        [B, D, H, W, 3], in [-1,1]
        appearance:  [B, 1, D, H, W] (optional if self.appearance=False)
        """
        dtype = latent_code.dtype
        device = latent_code.device
        grid = grid.to(device=device, dtype=dtype)
        grid = torch.clamp(grid, -1.0, 1.0)

        # Expand latent z to (B,C,1,1,1)
        x = latent_code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # multi-scale volumetric features
        f1 = self.decoder_1(x)
        f2 = self.decoder_2(f1)
        f3 = self.decoder_3(f2)
        f4 = self.decoder_4(f3)

        # sample features at query grid (grid_sample: for 5D input, mode='bilinear' == trilinear)
        s1 = F.grid_sample(f1, grid, mode="bilinear", align_corners=True)
        s2 = F.grid_sample(f2, grid, mode="bilinear", align_corners=True)
        s3 = F.grid_sample(f3, grid, mode="bilinear", align_corners=True)
        s4 = F.grid_sample(f4, grid, mode="bilinear", align_corners=True)

        # Reshape coords to [B,3,D,H,W]
        coords = grid.permute(0, 4, 1, 2, 3)
        if self.use_fourier:
            coords_feat = self.ff(coords)
        else:
            coords_feat = coords

        feats = [coords_feat, s1, s2, s3, s4]
        if self.appearance:
            if appearance is None:
                raise ValueError("appearance=True but appearance tensor is None")
            feats.append(appearance.to(device=device, dtype=dtype))

        feat = torch.cat(feats, dim=1)
        logits = self.implicit_mlp(feat)
        return logits

class EmbeddingDecoder(nn.Module):
    """
    Maintain trainable latent code z per-sample (nn.Embedding) and decode to K-class logits.
    
    Args:
        n_samples: Should match the index range of your Dataset (or leave sufficient upper bound)
    """
    def __init__(
        self,
        latent_dimension: int = 256,
        n_samples: int = 120,
        num_classes: int = 11,
        appearance: bool = True,
        use_fourier: bool = False,
        fourier_bands: int = 6,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Embedding(n_samples, latent_dimension)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)

        self.decoder = ImplicitDecoder(
            latent_dimension=latent_dimension,
            out_channels=num_classes,
            norm=DEFAULT["norm"],
            activation=DEFAULT["activation"],
            appearance=appearance,
            use_fourier=use_fourier,
            fourier_bands=fourier_bands,
            dropout_p=dropout_p,
        )

    def forward(self, indices, grid, appearance):
        """
        Args:
            indices: [B] long (one embedding id per sample)
            grid: [B, D, H, W, 3]
            appearance: [B, 1, D, H, W]
        
        Returns:
            logits: [B,K,D,H,W]
            z: [B,C] latent codes
        """
        z = self.encoder(indices)
        logits = self.decoder(z, grid, appearance)
        return logits, z
