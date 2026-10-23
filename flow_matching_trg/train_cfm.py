"""Train the conditional flow matching velocity model v_theta(u_s, s, u0).

    python train_cfm.py            # paper configuration (configs/cfm.yaml)
    python train_cfm.py --dummy    # two epochs on the bundled sample data

Writes best.pt (lowest validation loss), latest.pt and TensorBoard logs to ``output_dir``.
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
from trg.velocity_model import VelocityModel


def run_epoch(model, loader, device, optimizer=None, grad_clip=None):
    model.train(optimizer is not None)
    losses = []
    with torch.set_grad_enabled(optimizer is not None):
        for x0, x1 in loader:
            loss = model.compute_loss(x0.to(device), x1.to(device))
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.item())
    return float(np.mean(losses))


def main():
    cfg = load_config("cfm", __doc__)
    tc = cfg.training
    set_seed(tc.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.output_dir)
    os.makedirs(out, exist_ok=True)

    model = VelocityModel(base_channels=cfg.model.base_channels).to(device)
    print(f"Velocity model parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_loader = make_loader(cfg, "train", tc.seed)
    val_loader = make_loader(cfg, "test", tc.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tc.plateau_factor, patience=tc.plateau_patience, min_lr=tc.min_lr)
    writer = SummaryWriter(log_dir=str(out / "logs"))

    best, stale = float("inf"), 0
    for epoch in range(tc.max_epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer, tc.grad_clip)
        val_loss = run_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)

        ckpt = {"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_loss": min(best, val_loss),
                "config": OmegaConf.to_container(cfg)}
        torch.save(ckpt, out / "latest.pt")
        status = ""
        if val_loss < best:
            best, stale, status = val_loss, 0, "best"
            torch.save(ckpt, out / "best.pt")
        else:
            stale += 1
        print(f"epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {lr:.2e} {status}", flush=True)
        if stale >= tc.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    writer.close()


if __name__ == "__main__":
    main()
