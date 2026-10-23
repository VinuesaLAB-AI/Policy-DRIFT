"""Configuration loading shared by train_cfm.py and train_trg.py."""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_config(name, description):
    """Load configs/<name>.yaml merged with the dataset config.

    ``--dummy`` swaps in configs/dummy.yaml (bundled sample data, two epochs).
    Any further ``key=value`` arguments override single entries, e.g. ``training.batch_size=64``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dummy", action="store_true", help="short run on the bundled sample data")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load(CONFIG_DIR / "dataset.yaml"), OmegaConf.load(CONFIG_DIR / f"{name}.yaml"))
    if args.dummy:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(CONFIG_DIR / "dummy.yaml"))
        cfg.output_dir = f"{cfg.output_dir}_dummy"
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    resolve = lambda p: str(p if Path(p).is_absolute() else REPO_ROOT / p)
    cfg.dataset.regimes = {k: resolve(v) for k, v in cfg.dataset.regimes.items() if v is not None}
    cfg.dataset.mean_file = resolve(cfg.dataset.mean_file)
    cfg.output_dir = resolve(cfg.output_dir)
    if "cfm_checkpoint" in cfg:
        cfg.cfm_checkpoint = resolve(cfg.cfm_checkpoint)
    return cfg


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
