"""Snapshot-pair dataset (u0, u1) separated by Delta t+ = 5.

Each regime directory holds snapshots_s<N>.h5 files with datasets u, v, w of shape (1, 16, 16, 64)
and time of shape (1,). Snapshots are sorted by time, the first ``max_snapshots`` are kept and each
regime is split chronologically into train (first ``train_split``) and test (rest). Pairs never
cross regime boundaries. Fields are turned into fluctuations by subtracting the per-voxel mean
and min-max normalised per channel.
"""

import functools
import glob
import os
import re

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_regime(directory, max_snapshots=None, dtype=torch.float32):
    files = sorted(glob.glob(os.path.join(directory, "snapshots_s*.h5")),
                   key=lambda p: int(re.search(r"snapshots_s(\d+)\.h5$", p).group(1)))
    if max_snapshots is not None:
        files = files[:max_snapshots]
    if not files:
        raise FileNotFoundError(f"No snapshots_s*.h5 files in {directory}")
    data, times = [], []
    for path in files:
        with h5py.File(path, "r") as f:
            data.append(np.stack([f["u"][0], f["v"][0], f["w"][0]], axis=0))
            times.append(float(f["time"][0]))
    order = np.argsort(np.array(times))
    return torch.from_numpy(np.stack(data, axis=0)[order]).to(dtype)


def load_mean(mean_file):
    with h5py.File(mean_file, "r") as f:
        return torch.from_numpy(np.stack([f["u_mean"][:], f["v_mean"][:], f["w_mean"][:]], axis=0)).float()


def normalize_minmax(x, mins, maxs, eps=1e-9):
    mins_ = torch.as_tensor(mins, dtype=x.dtype).view(1, -1, 1, 1, 1)
    maxs_ = torch.as_tensor(maxs, dtype=x.dtype).view(1, -1, 1, 1, 1)
    return (x - mins_) / (maxs_ - mins_ + eps)


class SnapshotPairs(Dataset):
    def __init__(self, regimes, split, mean_file, mins, maxs, train_split=0.8, max_snapshots=None):
        assert split in ("train", "test")
        mean = load_mean(mean_file)
        self.sources, self.index = [], []
        for directory in regimes:
            data = load_regime(directory, max_snapshots)
            n_train = int(train_split * data.shape[0])
            data = data[:n_train] if split == "train" else data[n_train:]
            data = normalize_minmax(data - mean.unsqueeze(0).to(dtype=data.dtype), mins, maxs)
            self.index += [(len(self.sources), i) for i in range(len(data) - 1)]
            self.sources.append(data)
            print(f"  {split}: {len(data) - 1} pairs from {directory}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        src, i = self.index[idx]
        return self.sources[src][i], self.sources[src][i + 1]


def _worker_init(worker_id, seed):
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def make_loader(cfg, split, seed):
    """Training loader over all regimes; test loader over the first (uncontrolled) regime."""
    ds = cfg.dataset
    regimes = list(ds.regimes.values()) if split == "train" else [list(ds.regimes.values())[0]]
    dataset = SnapshotPairs(regimes, split, ds.mean_file, ds.mins, ds.maxs,
                            train_split=ds.train_split, max_snapshots=ds.max_snapshots)
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = cfg.training.num_workers
    return DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      worker_init_fn=functools.partial(_worker_init, seed=seed) if workers > 0 else None,
                      generator=generator, drop_last=(split == "train"))
