"""Drag reduction and actuation energy of an evaluation trajectory, as reported in the paper.

DR    = 1 - <du/dy|_wall>_xz / 7.9
E_act = 0.5 <|v_w|^3>_xz / u_tau^3
Both are averaged over t+ in [500, 3000], with t+ = DNS iteration * 0.5 / 13.
"""

from pathlib import Path

import h5py
import numpy as np

DT_PLUS = 0.5 / 13
WINDOW = (500.0, 3000.0)
WALL_SHEAR_REF = 7.9
U_TAU = 0.04287


def _window(f):
    t = np.array(f["scales/iteration"]).ravel() * DT_PLUS
    return (t >= WINDOW[0]) & (t <= WINDOW[1])


def trajectory_summary(traj_dir):
    traj_dir = Path(traj_dir)
    with h5py.File(traj_dir / "scalars" / "scalars_s1.h5", "r") as f:
        dr = (1.0 - np.array(f["tasks/dudy_wall"]).ravel() / WALL_SHEAR_REF)[_window(f)].mean()
    with h5py.File(traj_dir / "plane_bc" / "plane_bc_s1.h5", "r") as f:
        v = np.array(f["tasks/v_bc"]).squeeze(-1)
        energy = (0.5 * np.mean(np.abs(v) ** 3, axis=(1, 2)) / U_TAU ** 3)[_window(f)].mean()
    return {"DR": float(dr), "E_act": float(energy)}
