"""Run a trained policy deterministically from held-out initial conditions.

    python evaluate.py --model ../checkpoints/drl/policy_drift_td3.zip
    python evaluate.py --model ../checkpoints/drl/td3_wse.zip --inits 1 --n-outputs 160

For every restart file data/eval/snapshots_s<i>.h5 (i in --inits, default 1..6) the DNS is run for
``n_outputs`` actions (t+ = 3027 by default) and its outputs are written to
outputs/evaluation/<model name>/init_<i>/{scalars,plane_bc,planes,profiles}. The drag reduction
and actuation energy averaged over t+ in [500, 3000] are printed; summary.json collects all
finished trajectories in the output directory.
"""

import argparse
import json
import logging
import shutil
import zipfile
from pathlib import Path

import numpy as np
from dedalus.tools import post
from stable_baselines3 import SAC, TD3

from drl import config, metrics
from drl.env import ChannelEnv


def load_policy(path):
    module = json.loads(zipfile.ZipFile(path).read("data"))["policy_class"]["__module__"]
    return (SAC if ".sac." in module else TD3).load(path, device="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Stable-Baselines3 .zip checkpoint")
    parser.add_argument("--ics", default=str(config.DATA_DIR / "eval"), help="directory with snapshots_s<i>.h5")
    parser.add_argument("--inits", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="initial conditions to run")
    parser.add_argument("--n-outputs", type=int, default=None, help="actions per trajectory (default 4920)")
    parser.add_argument("--out", default=None, help="output directory (default outputs/evaluation/<model name>)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = config.load()
    cfg["run"] = {"name": "evaluation", "reward": "wse"}
    n_actions = args.n_outputs or cfg["evaluation"]["n_outputs"]
    out = Path(args.out) if args.out else config.DRL_ROOT / "outputs" / "evaluation" / Path(args.model).stem
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.yml"
    config.save(cfg, config_path)
    model = load_policy(args.model)

    for i in args.inits:
        traj_dir = out / f"init_{i}"
        if traj_dir.exists():
            shutil.rmtree(traj_dir)
        env = ChannelEnv(cfg, config_path, "evaluate", traj_dir, restart_file=Path(args.ics) / f"snapshots_s{i}.h5")
        observations, _ = env.reset()
        for step in range(n_actions):
            state = np.array([observations[a] for a in env.possible_agents])
            action, _ = model.predict(state, deterministic=True)
            action = action - np.mean(action)
            observations, *_ = env.step(dict(zip(env.possible_agents, action)))
            if step % 130 == 0:
                print(f"init_{i}: action {step}/{n_actions}", flush=True)
        env.close()
        for name in ("scalars", "plane_bc", "planes", "profiles"):
            post.merge_process_files(str(traj_dir / name), cleanup=True)
        print(f"init_{i}: {metrics.trajectory_summary(traj_dir)}", flush=True)

    summary = {d.name: metrics.trajectory_summary(d) for d in sorted(out.glob("init_*"))
               if (d / "plane_bc" / "plane_bc_s1.h5").exists()}
    summary["mean"] = {k: float(np.mean([s[k] for s in summary.values()])) for k in ("DR", "E_act")}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["mean"]))


if __name__ == "__main__":
    main()
