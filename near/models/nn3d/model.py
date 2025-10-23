import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import LatentCodeUpsample, ConvNormAct

DEFAULT = {
    "norm": lambda c: nn.GroupNorm(8, c),
    "activation": nn.LeakyReLU
}

class ImplicitDecoder(nn.Module):
    """Decode latent z → multi-scale 3D features; sample at grid; point-MLP → K logits."""
    def __init__(self, latent_dimension, out_channels, norm, activation,
                 decoder_channels=[64,48,32,16], appearance=True):
        super().__init__()
        self.appearance = appearance
        # pyramid from (B,C,1,1,1)
        self.decoder_1 = nn.Sequential(
            LatentCodeUpsample(latent_dimension, upsample_factor=2, channel_reduction=2, norm=None, activation=activation),
            LatentCodeUpsample(latent_dimension // 2, upsample_factor=2, channel_reduction=2, norm=norm, activation=activation),
            ConvNormAct(latent_dimension // 4, decoder_channels[0], norm=norm, activation=activation)
        )
        self.decoder_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[0], decoder_channels[1], norm=norm, activation=activation),
            ConvNormAct(decoder_channels[1], decoder_channels[1], norm=norm, activation=activation)
        )
        self.decoder_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[1], decoder_channels[2], norm=norm, activation=activation),
            ConvNormAct(decoder_channels[2], decoder_channels[2], norm=norm, activation=activation)
        )
        self.decoder_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvNormAct(decoder_channels[2], decoder_channels[3], norm=norm, activation=activation),
            ConvNormAct(decoder_channels[3], decoder_channels[3], norm=norm, activation=activation)
        )
        in_ch = 3 + sum(decoder_channels) + (1 if appearance else 0)
        self.implicit_mlp = nn.Sequential(
            nn.Conv3d(in_ch, 64, 1), norm(64), activation(),
            nn.Conv3d(64, 32, 1),  norm(32), activation(),
            nn.Conv3d(32, out_channels, 1)  # logits (no softmax)
        )

    def forward(self, latent_code, grid, appearance):
        # expand z to (B,C,1,1,1)
        x = latent_code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # multi-scale features
        f1 = self.decoder_1(x)
        f2 = self.decoder_2(f1)
        f3 = self.decoder_3(f2)
        f4 = self.decoder_4(f3)
        # sample features at grid
        s1 = F.grid_sample(f1, grid, mode="bilinear", align_corners=True)
        s2 = F.grid_sample(f2, grid, mode="bilinear", align_corners=True)
        s3 = F.grid_sample(f3, grid, mode="bilinear", align_corners=True)
        s4 = F.grid_sample(f4, grid, mode="bilinear", align_corners=True)
        coords = grid.permute(0,4,1,2,3)
        feat = torch.cat([coords, s1, s2, s3, s4, appearance] if self.appearance else [coords, s1, s2, s3, s4], dim=1)
        return self.implicit_mlp(feat)

class EmbeddingDecoder(nn.Module):
    """Maintain trainable z per-sample (nn.Embedding) and decode to K-class logits."""
    def __init__(self, latent_dimension=256, n_samples=120, num_classes=11, appearance=True):
        super().__init__()
        self.encoder = nn.Embedding(n_samples, latent_dimension)
        self.decoder = ImplicitDecoder(
            latent_dimension=latent_dimension,
            out_channels=num_classes,
            norm=DEFAULT["norm"],
            activation=DEFAULT["activation"],
            appearance=appearance
        )

    def forward(self, indices, grid, appearance):
        z = self.encoder(indices)
        logits = self.decoder(z, grid, appearance)
        return logits, z