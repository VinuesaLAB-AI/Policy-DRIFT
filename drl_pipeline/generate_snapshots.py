"""Generate flow-field snapshots for the CFM training data.

    python generate_snapshots.py --control none --out ../data/cfm/uncontrolled
    python generate_snapshots.py --control opposition --out ../data/cfm/opposition_control
    python generate_snapshots.py --control ../checkpoints/drl/td3_wse.zip --out ../data/cfm/td3_wse

Every run starts from data/generation/snapshot_restart.h5 and saves u, v, w and p every 130 DNS steps
(Delta t+ = 5) as <out>/snapshots_s<k>.h5, k = 1..n-snapshots, the first file being the initial field.
The wall actuation is

    none        zero (uncontrolled flow)
    opposition  v_w = -(v - <v>_xz) at y+ = 15, updated every DNS step by the DNS workers
    <policy>    the actions of a trained Stable-Baselines3 policy, updated every action (16 DNS steps)

Needs nprocx * nprocz + 1 MPI slots (65 by default).
"""

import argparse
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
from dedalus.tools import post

from drl import config
from drl.env import ChannelEnv
from evaluate import load_policy


def controller(control, cfg):
    """Maps the observations [n_agents, (v'+, u'+)] to actions in units of max_amplitude.

    Opposition control is applied by the DNS workers themselves, so the agent sends zero actions.
    """
    if control in ("none", "opposition"):
        return lambda obs: np.zeros((len(obs), 1), dtype=np.float32)
    model = load_policy(control)

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action - np.mean(action)
    return policy


def write_snapshots(merged, out):
    """Split the merged Dedalus output into one file per snapshot, in the CFM data format."""
    with h5py.File(merged, "r") as f:
        coords = {c: f["scales"][c][list(f["scales"][c].keys())[0]][:] for c in "xyz"}
        time = np.array(f["scales/sim_time"]).ravel()
        iteration = np.array(f["scales/iteration"]).ravel()
        for k in range(len(time)):
            with h5py.File(out / f"snapshots_s{k + 1}.h5", "w") as g:
                for name in ("u", "v", "w", "p"):
                    g.create_dataset(name, data=f["tasks"][name][k:k + 1].astype(np.float32))
                for c, values in coords.items():
                    g.create_dataset(c, data=values[np.newaxis, :].astype(np.float32))
                g.create_dataset("time", data=np.array([time[k] - time[0]], dtype=np.float32))
                g.attrs["iteration"] = int(iteration[k] - iteration[0])
    return len(time)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", required=True, help="none, opposition or a policy .zip")
    parser.add_argument("--n-snapshots", type=int, default=7350)
    parser.add_argument("--out", required=True, help="output directory for snapshots_s<k>.h5")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = config.load()
    cfg["run"] = {"name": "generate", "reward": "wse", "control": args.control}
    out = Path(args.out).resolve()
    work = out / "dedalus"
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "config.yml"
    config.save(cfg, config_path)

    act = controller(args.control, cfg)
    env = ChannelEnv(cfg, config_path, "generate", work,
                     restart_file=config.DATA_DIR / "generation" / "snapshot_restart.h5")
    last_write = (args.n_snapshots - 1) * cfg["simulation"]["snapshot_every"]
    n_actions = last_write // config.dns_steps_per_action(cfg) + 1

    observations, _ = env.reset()
    for step in range(n_actions):
        state = np.array([observations[a] for a in env.possible_agents])
        observations, *_ = env.step(dict(zip(env.possible_agents, act(state))))
        if step % 130 == 0:
            print(f"action {step}/{n_actions}", flush=True)
    env.close()

    for name in ("state", "scalars"):
        post.merge_process_files(str(work / name), cleanup=True)
    n = write_snapshots(work / "state" / "state_s1.h5", out)
    shutil.move(work / "scalars" / "scalars_s1.h5", out / "scalars.h5")
    shutil.rmtree(work)
    print(f"Wrote {n} snapshots to {out}")


if __name__ == "__main__":
    main()
