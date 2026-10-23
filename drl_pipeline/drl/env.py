"""Multi-agent channel-flow environment: one agent per wall grid point, all sharing the policy.

Each agent observes (v'+, u'+) at y+ = 15 above its wall point and sets the local wall-normal
velocity. Rewards (identical for all agents, averaged over the 16 DNS steps of an action):

    wse  DR = 1 - tau_w / tau_w,0
    wen  DR - E_act,  E_act = 0.5 <|v_w|^3>_xz / u_tau^3
    trg  -RMSE between the flow field and the TRG target (train mode); DR in validation mode

Modes: ``train`` starts every episode from one of the six fields in data/restart (drawn with a
seeded generator), ``validation`` from data/validation, ``evaluate`` and ``generate`` from a given
restart file.
"""

import functools
import logging
from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np
from pettingzoo import ParallelEnv

from drl import config
from drl.solver_io import SolverIO

logger = logging.getLogger(__name__)


class ChannelEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "channel_flow"}

    def __init__(self, cfg, config_path, mode, out_dir, restart_file=None, seed=0):
        self.cfg = cfg
        self.mode = mode
        self.reward_type = cfg["run"]["reward"]
        self.out_dir = Path(out_dir)
        self.io = SolverIO(cfg, config_path)
        self.possible_agents = self.io.agents
        self.agents = self.possible_agents[:]
        self.steps_per_action = config.dns_steps_per_action(cfg)
        self.restart_files = {
            "train": sorted((config.DATA_DIR / "restart").glob("snapshots_s*.h5")),
            "validation": [config.DATA_DIR / "validation" / "snapshot_validation.h5"],
            "evaluate": [restart_file],
            "generate": [restart_file],
        }[mode]
        self.rng = np.random.default_rng(seed)
        self.render_mode = None
        self.comm = None
        self.n_resets = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-np.inf, high=np.inf, shape=[2], dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=[1], dtype=np.float32)

    def close(self):
        if self.comm is not None:
            self.io.stop(self.comm)
            self.comm = None

    def reset(self, seed=None, options=None):
        self.close()
        restart = self.restart_files[self.rng.integers(len(self.restart_files))]
        self.n_resets += 1
        self.n_actions = 0
        logger.info(f"Reset {self.n_resets}: {restart}")
        self.agents = self.possible_agents[:]
        self.comm = self.io.start(self.mode, restart, self.out_dir)
        _, observations = self.io.state(self.comm)
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        self.io.state(self.comm)
        self.io.actuate(self.comm, actions)
        self.n_actions += 1
        reward, dr = self._evolve()
        _, observations = self.io.state(self.comm)

        done = self.n_actions >= self.cfg["training"]["episode_length"]
        if self.mode == "train" and dr > self.cfg["reward"]["relaminarisation_dr"]:
            done = True
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        return observations, rewards, terminations, truncations, {agent: {} for agent in self.agents}

    def _evolve(self):
        wen = self.reward_type == "wen" and self.mode in ("train", "validation")
        tracking = self.reward_type == "trg" and self.mode == "train"
        terms = self.io.evolve(self.comm, self.steps_per_action, energy=wen, tracking=tracking)
        dr_vec = 1 - terms["shear"] / self.cfg["reward"]["wall_shear_ref"]
        if wen:
            u_tau3 = self.cfg["control"]["max_amplitude"] ** 3
            reward_vec = dr_vec - terms["energy"] / u_tau3
        elif tracking:
            reward_vec = terms["tracking"]
        else:
            reward_vec = dr_vec
        reward, dr = float(np.mean(reward_vec)), float(np.mean(dr_vec))
        logger.info(f"Action {self.n_actions}: reward {reward:.4f}, DR {100 * dr:.1f}%")
        return reward, dr
