"""Conditional flow matching (CFM) velocity model v_theta(u_s, s, u0).

The probability path is u_s = (1 - s) * eta + s * u1 with eta ~ N(0, I), so the
regression target is the constant velocity u1 - eta. The network sees
[u_s, u0, s] concatenated along channels (7 input channels).
"""

import torch
import torch.nn as nn

from trg.unet import UNet3D


class VelocityModel(nn.Module):
    def __init__(self, n_channels=3, base_channels=64, kernel_size=3):
        super().__init__()
        self.unet = UNet3D(in_channels=2 * n_channels + 1, out_channels=n_channels,
                           base_channels=base_channels, kernel_size=kernel_size)

    def forward(self, x_s, x0, s):
        s_channel = s.view(-1, 1, 1, 1, 1).expand(-1, 1, *x_s.shape[2:])
        return self.unet(torch.cat([x_s, x0, s_channel], dim=1))

    def compute_loss(self, x0, x1):
        eta = torch.randn_like(x1)
        s = torch.rand(x1.shape[0], device=x1.device)
        s_ = s.view(-1, *[1] * (x1.dim() - 1))
        x_s = (1 - s_) * eta + s_ * x1
        return (self(x_s, x0, s) - (x1 - eta)).pow(2).mean()

    def integrate(self, x0, x_current, s=0.0, ds=1.0, steps=50):
        """Explicit Euler integration of du/ds = v_theta from s to s + ds."""
        device = x0.device
        s = torch.tensor(s, device=device)
        ds = torch.tensor(ds, device=device)
        steps = torch.tensor(steps, device=device)
        ds_small = ((1. * ds) / steps).expand(len(x0))
        ds_small_ = ds_small.view(-1, *[1] * (x0.dim() - 1))
        x = x_current
        for i in range(steps):
            x = x + ds_small_ * self(x, x0, s + i * ds_small)
        return x

    def sample(self, x0, eta=None, steps=50):
        if eta is None:
            eta = torch.randn_like(x0)
        return self.integrate(x0, eta, s=0.0, ds=1.0, steps=steps)
