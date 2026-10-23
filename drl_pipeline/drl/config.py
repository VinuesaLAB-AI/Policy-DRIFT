"""Run configuration shared by the agent process and the spawned DNS workers."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DRL_ROOT = REPO_ROOT / "drl_pipeline"
DATA_DIR = REPO_ROOT / "data"


def load(path=DRL_ROOT / "configs" / "channel.yml"):
    with open(path) as f:
        return yaml.safe_load(f)


def save(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def dns_steps_per_action(cfg):
    """DNS steps between two actions."""
    return int(cfg["simulation"]["delta_t"] / cfg["simulation"]["dt"])
