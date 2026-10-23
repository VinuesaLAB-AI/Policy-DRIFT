"""Table 1 statistics from evaluation outputs.

    python scripts/table1.py drl_pipeline/outputs/evaluation/*

For every evaluation directory (one init_<i> folder per initial condition) prints the drag
reduction in percent and E_act x 10, as mean and standard deviation over the trajectories, and
R = DR - E_act from the means.
"""

import sys
from pathlib import Path

import numpy as np

from drl.metrics import trajectory_summary


def main(directories):
    print(f"{'run':28s} {'DR mean':>8s} {'DR std':>7s} {'E mean':>8s} {'E std':>7s} {'R':>6s}")
    for d in map(Path, directories):
        runs = [trajectory_summary(t) for t in sorted(d.glob("init_*"))]
        if not runs:
            continue
        dr = np.array([r["DR"] for r in runs])
        e = np.array([r["E_act"] for r in runs])
        print(f"{d.name:28s} {100 * dr.mean():8.2f} {100 * dr.std():7.2f} {10 * e.mean():8.4f} {10 * e.std():7.4f}"
              f" {dr.mean() - e.mean():6.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
