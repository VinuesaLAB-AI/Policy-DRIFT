"""Train the reward predictor R_psi used by Terminal Reward Guidance.

For every batch of conditioning snapshots u0, an unguided ODE rollout of the frozen CFM model gives
the states u_{s_i} (i = 0..T-1) and the terminal state u1. R_psi(u_{s_i}, s_i) is regressed onto
r_T = R(u1) with the ramp weight w_i = s_i:

    L = sum_i (s_i / T) * 0.5 * (R_psi(u_{s_i}, s_i) - r_T)^2

    python train_trg.py            # paper configuration (configs/trg.yaml)
    python train_trg.py --dummy    # two epochs on the bundled sample data

The checkpoint with the lowest mean |R_psi(u1, s_{T-1}) - r_T| over the training epoch is saved
as best.pt; validation reports the reward of unguided rollouts on the test split.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from trg.config import load_config, set_seed
from trg.data import make_loader
from trg.reward import RewardFunction
from trg.reward_predictor import RewardPredictor
from trg.sampler import FlowStep, rollout, s_grid
from trg.velocity_model import VelocityModel


def train_epoch(predictor, flow_step, reward_fn, loader, optimizer, cfg, device):
    predictor.train()
    grid = s_grid(cfg.T)
    losses, errors = [], []
    for x0, _ in loader:
        x0 = x0.to(device)
        B = x0.shape[0]
        trajectory = rollout(flow_step, x0, cfg.T)
        x_T = trajectory[-1].detach()
        with torch.no_grad():
            r_T, _, _ = reward_fn(x_T)

        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for i in range(cfg.T):
            s_batch = torch.full((B,), grid[i], device=device, dtype=x0.dtype)
            r_pred = predictor(trajectory[i].detach(), s_batch)
            loss = loss + (grid[i] / cfg.T) * 0.5 * (r_pred - r_T).pow(2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), cfg.training.grad_clip)
        optimizer.step()

        with torch.no_grad():
            s_last = torch.full((B,), grid[-2], device=device, dtype=x0.dtype)
            errors.append((predictor(x_T, s_last) - r_T).abs().mean().item())
        losses.append(loss.item())
    return float(np.mean(losses)), float(np.mean(errors))


@torch.no_grad()
def validate(flow_step, reward_fn, loader, T, device):
    rewards, drs, energies = [], [], []
    for x0, _ in loader:
        r, dr, e = reward_fn(rollout(flow_step, x0.to(device), T)[-1])
        rewards.append(r.mean().item())
        drs.append(dr.mean().item())
        energies.append(e.mean().item())
    return float(np.mean(rewards)), float(np.mean(drs)), float(np.mean(energies))


def main():
    cfg = load_config("trg", __doc__)
    tc = cfg.training
    set_seed(tc.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.output_dir)
    os.makedirs(out, exist_ok=True)

    velocity_model = VelocityModel().to(device)
    velocity_model.load_state_dict(torch.load(cfg.cfm_checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    flow_step = FlowStep(velocity_model, n_euler_steps=cfg.n_euler_steps, T=cfg.T)
    reward_fn = RewardFunction(alpha=cfg.alpha, **cfg.dataset.reward).to(device)
    predictor = RewardPredictor(hidden_channels=cfg.hidden_channels).to(device)
    print(f"Reward predictor parameters: {sum(p.numel() for p in predictor.parameters()):,}")

    train_loader = make_loader(cfg, "train", tc.seed)
    val_loader = make_loader(cfg, "test", tc.seed)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc.max_epochs)
    writer = SummaryWriter(log_dir=str(out / "logs"))

    best, stale = float("inf"), 0
    for epoch in range(tc.max_epochs):
        loss, err = train_epoch(predictor, flow_step, reward_fn, train_loader, optimizer, cfg, device)
        scheduler.step()
        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("train/r_pred_err", err, epoch)
        line = f"epoch {epoch:4d} | loss {loss:.4e} | r_pred_err {err:.4e}"
        if (epoch + 1) % tc.val_every_n_epochs == 0:
            reward, dr, energy = validate(flow_step, reward_fn, val_loader, cfg.T, device)
            for k, v in {"reward": reward, "DR": dr, "energy": energy}.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            line += f" | val reward {reward:.4f} DR {dr:.4f} E {energy:.4f}"

        ckpt = {"model_state_dict": predictor.state_dict(), "epoch": epoch, "best_r_pred_err": min(best, err),
                "config": OmegaConf.to_container(cfg)}
        torch.save(ckpt, out / "latest.pt")
        if err < best:
            best, stale = err, 0
            torch.save(ckpt, out / "best.pt")
            line += " best"
        else:
            stale += 1
        print(line, flush=True)
        if stale >= tc.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    writer.close()


if __name__ == "__main__":
    main()
