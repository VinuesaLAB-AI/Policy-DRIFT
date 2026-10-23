"""Reward predictor R_psi(u_s, s): predicts the terminal reward from an intermediate ODE state.

Four Conv3d -> GroupNorm(8) -> FiLM(s) -> SiLU stages (the second one with stride 2),
followed by a 1x1x1 head whose 8 x 8 x 32 score map is summed into a scalar.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(s, d_emb=64):
    half = d_emb // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=s.device, dtype=s.dtype) / half)
    args = s[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ConvStage(nn.Module):
    def __init__(self, c_in, c_out, stride, d_emb):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(min(8, c_out), c_out)
        self.film = nn.Linear(d_emb, 2 * c_out)
        self.act = nn.SiLU()

    def forward(self, x, s_emb):
        h = self.norm(self.conv(x))
        gamma, beta = self.film(s_emb).chunk(2, dim=1)
        h = gamma.view(-1, h.shape[1], 1, 1, 1) * h + beta.view(-1, h.shape[1], 1, 1, 1)
        return self.act(h)


class RewardPredictor(nn.Module):
    def __init__(self, n_channels=3, hidden_channels=64, d_emb=64):
        super().__init__()
        h = hidden_channels
        self.d_emb = d_emb
        self.stage0 = ConvStage(n_channels, h, stride=1, d_emb=d_emb)
        self.stage1 = ConvStage(h, h, stride=2, d_emb=d_emb)
        self.stage2 = ConvStage(h, 2 * h, stride=1, d_emb=d_emb)
        self.stage3 = ConvStage(2 * h, 4 * h, stride=1, d_emb=d_emb)
        self.head = nn.Conv3d(4 * h, 1, kernel_size=1)
        with torch.no_grad():
            self.head.weight.div_(8 * 8 * 32)
            self.head.bias.zero_()

    def forward(self, x_s, s):
        s_emb = sinusoidal_embedding(s, self.d_emb)
        h = self.stage0(x_s, s_emb)
        h = self.stage1(h, s_emb)
        h = self.stage2(h, s_emb)
        h = self.stage3(h, s_emb)
        return self.head(h).flatten(1).sum(1)
