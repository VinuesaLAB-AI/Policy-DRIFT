"""Terminal reward R(u1) = DR(u1) - alpha * E(u1), evaluated on min-max normalised fluctuation fields.

DR: mean wall shear from the first two wall-normal points, shifted by the contribution of the mean
    profile (du_dy_zero_offset) and divided by the uncontrolled reference wall shear (7.9,
    expressed in normalised units as du_dy_ref).
E:  0.5 * <|v_w - v_zero_offset|^3>_xz / v_ref, where v_zero_offset is the normalised value of
    v' = 0 and v_ref scales E to about 0.5 for controlled snapshots of the training data.
Both terms are capped at 1. The constants are set in configs/dataset.yaml.
"""

import torch
import torch.nn as nn


class RewardFunction(nn.Module):
    def __init__(self, dy, du_dy_zero_offset, du_dy_ref, v_zero_offset, v_ref, alpha=1.0):
        super().__init__()
        self.register_buffer("du_dy_ref", torch.tensor(du_dy_ref, dtype=torch.float32))
        self.register_buffer("du_dy_zero_offset", torch.tensor(du_dy_zero_offset, dtype=torch.float32))
        self.register_buffer("v_ref", torch.tensor(v_ref, dtype=torch.float32))
        self.register_buffer("dy", torch.tensor(dy, dtype=torch.float32))
        self.register_buffer("v_zero_offset", torch.tensor(v_zero_offset, dtype=torch.float32))
        self.alpha = alpha

    def drag_reduction(self, x):
        u = x[:, 0]
        mean_shear = ((u[..., 1] - u[..., 0]) / self.dy).mean(dim=(1, 2))
        return 1.0 - (mean_shear + self.du_dy_zero_offset) / self.du_dy_ref

    def energy(self, x):
        v_wall = x[:, 1, :, :, 0]
        return 1 / 2 * (((v_wall - self.v_zero_offset).abs()) ** 3).mean(dim=(1, 2)) / self.v_ref

    def forward(self, x):
        """Returns (reward, drag reduction, energy), each of shape [B]."""
        dr = self.drag_reduction(x).clamp(max=1.0)
        energy = self.energy(x).clamp(max=1.0)
        return dr - self.alpha * energy, dr, energy
