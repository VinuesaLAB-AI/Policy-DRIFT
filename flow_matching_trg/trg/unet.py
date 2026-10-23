"""3D U-Net used as the backbone of the CFM velocity model.

Input and output volumes are [B, C, X, Z, Y] = [B, C, 16, 16, 64].
Encoder: four levels (base, 2x, 4x, 8x channels) of two Conv3d -> BatchNorm3d -> ReLU
blocks, with 2x average pooling after the first three levels.
Decoder: transposed-convolution upsampling, skip concatenation and one conv block per level.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_normal(shape, fan_in):
    return np.sqrt(1 / fan_in) * torch.randn(*shape)


class Conv3d(nn.Module):
    """Same-padded 3D convolution with optional 2x transposed upsampling."""

    def __init__(self, in_channels, out_channels, kernel, up=False):
        super().__init__()
        self.in_channels = in_channels
        self.up = up
        fan_in = in_channels * kernel ** 3
        self.weight = nn.Parameter(kaiming_normal([out_channels, in_channels, kernel, kernel, kernel], fan_in))
        self.bias = nn.Parameter(kaiming_normal([out_channels], fan_in) * 0.0)
        f = torch.ones(2)
        f3d = (f[:, None, None] * f[None, :, None] * f[None, None, :]).unsqueeze(0).unsqueeze(1) / f.sum().pow(3)
        self.register_buffer("resample_filter", f3d if up else None)

    def forward(self, x):
        w = self.weight.to(x.dtype)
        if self.up:
            f = self.resample_filter.to(x.dtype)
            x = F.conv_transpose3d(x, f.mul(8).tile([self.in_channels, 1, 1, 1, 1]),
                                   groups=self.in_channels, stride=2, padding=0)
        x = F.conv3d(x, w, padding=w.shape[-1] // 2)
        return x.add_(self.bias.to(x.dtype).reshape(1, -1, 1, 1, 1))


class BatchNorm3d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.BatchNorm3d(num_channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        return self.norm(x.float()).to(x.dtype)


class Activation(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def conv_block(in_c, out_c, kernel=3, activation=True, up=False):
    """Conv3d -> BatchNorm3d -> ReLU (identity activation when ``activation`` is False)."""
    act = Activation(F.relu) if activation else nn.Identity()
    return nn.Sequential(Conv3d(in_c, out_c, kernel=kernel, up=up), BatchNorm3d(out_c), act)


def crop_to_match(x, target):
    _, _, d_t, h_t, w_t = target.shape
    _, _, d_x, h_x, w_x = x.shape
    if d_x > d_t or h_x > h_t or w_x > w_t:
        x = x[:, :, :d_t, :h_t, :w_t]
    elif d_x < d_t or h_x < h_t or w_x < w_t:
        x = F.pad(x, [0, w_t - w_x, 0, h_t - h_x, 0, d_t - d_x])
    return x


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, kernel=3):
        super().__init__()
        n1, n2, n3, n4 = base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels
        self.enc1_conv1 = conv_block(in_channels, n1, kernel)
        self.enc1_conv2 = conv_block(n1, n1, kernel)
        self.pool1 = nn.AvgPool3d(2)
        self.enc2_conv1 = conv_block(n1, n2, kernel)
        self.enc2_conv2 = conv_block(n2, n2, kernel)
        self.pool2 = nn.AvgPool3d(2)
        self.enc3_conv1 = conv_block(n2, n3, kernel)
        self.enc3_conv2 = conv_block(n3, n3, kernel)
        self.pool3 = nn.AvgPool3d(2)
        self.enc4_conv1 = conv_block(n3, n4, kernel)
        self.enc4_conv2 = conv_block(n4, n4, kernel)

    def forward(self, x):
        s1 = self.enc1_conv2(self.enc1_conv1(x))
        s2 = self.enc2_conv2(self.enc2_conv1(self.pool1(s1)))
        s3 = self.enc3_conv2(self.enc3_conv1(self.pool2(s2)))
        bottleneck = self.enc4_conv2(self.enc4_conv1(self.pool3(s3)))
        return bottleneck, [s1, s2, s3]


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, kernel=3):
        super().__init__()
        n1, n2, n3, n4 = base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels
        self.dec3_up = conv_block(n4, n3, kernel, up=True)
        self.dec3_conv = conv_block(2 * n3, n3, kernel)
        self.dec2_up = conv_block(n3, n2, kernel, up=True)
        self.dec2_conv = conv_block(2 * n2, n2, kernel)
        self.dec1_up = conv_block(n2, n1, kernel, up=True)
        self.dec1_conv = conv_block(2 * n1, n1, kernel)
        self.output_conv = conv_block(n1, out_channels, kernel, activation=False)

    def forward(self, x, skips):
        s1, s2, s3 = skips
        x = self.dec3_conv(torch.cat([s3, crop_to_match(self.dec3_up(x), s3)], dim=1))
        x = self.dec2_conv(torch.cat([s2, crop_to_match(self.dec2_up(x), s2)], dim=1))
        x = self.dec1_conv(torch.cat([s1, crop_to_match(self.dec1_up(x), s1)], dim=1))
        return self.output_conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, kernel_size=3):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_channels, kernel_size)
        self.decoder = UNetDecoder(out_channels, base_channels, kernel_size)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        return self.decoder(bottleneck, skips)
