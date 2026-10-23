"""Compute the normalisation statistics in data/means.h5 and configs/dataset.yaml.

The per-voxel time mean and the per-channel min / max / std of the fluctuations are taken over
the first ``max_snapshots`` snapshots of every regime listed in configs/dataset.yaml.

    python compute_stats.py --out data/means.h5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from trg.config import CONFIG_DIR, REPO_ROOT
from trg.data import load_regime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/means.h5")
    args = parser.parse_args()

    ds = OmegaConf.load(CONFIG_DIR / "dataset.yaml").dataset
    data = np.concatenate([load_regime(str(REPO_ROOT / p), ds.max_snapshots, torch.float64).numpy()
                           for p in ds.regimes.values()], axis=0)
    mean = data.mean(axis=0)
    fluct = data - mean[None]

    with h5py.File(REPO_ROOT / args.out if not Path(args.out).is_absolute() else args.out, "w") as f:
        for c, name in enumerate("uvw"):
            f.create_dataset(f"{name}_mean", data=mean[c])
            f.create_dataset(f"{name}_std", data=fluct[:, c].std())
            f.create_dataset(f"{name}_min", data=float(fluct[:, c].min()))
            f.create_dataset(f"{name}_max", data=float(fluct[:, c].max()))
            print(f"{name}': min {fluct[:, c].min():.16g}  max {fluct[:, c].max():.16g}  std {fluct[:, c].std():.16g}")


if __name__ == "__main__":
    main()
