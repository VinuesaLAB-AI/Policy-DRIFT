"""Terminal Reward Guidance targets and the tracking reward used to train the Policy-DRIFT agent.

Every ``horizon`` actions the current DNS field u0 is normalised as in the CFM training data and
the TRG-guided CFM sample u1 (Delta t+ = 5 ahead) becomes the target. The reward at every DNS
step is the negative RMSE between the current normalised field and that target.
"""

import h5py
import numpy as np
import torch

from drl.config import repo_path
from trg.reward_predictor import RewardPredictor
from trg.sampler import FlowStep, TRGSampler
from trg.velocity_model import VelocityModel


class TRGTarget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.num_threads = cfg["num_threads"]
        with h5py.File(repo_path(cfg["means"]), "r") as f:
            self.mean = np.stack([f["u_mean"][:], f["v_mean"][:], f["w_mean"][:]], axis=0).astype(np.float32)
        self.mins = np.array(cfg["mins"])
        self.maxs = np.array(cfg["maxs"])

        velocity_model = VelocityModel()
        velocity_model.load_state_dict(self._weights(cfg["velocity_model"]))
        predictor = RewardPredictor(hidden_channels=cfg["hidden_channels"])
        predictor.load_state_dict(self._weights(cfg["reward_predictor"]))
        flow_step = FlowStep(velocity_model.to(self.device), n_euler_steps=cfg["n_euler_steps"], T=cfg["T"])
        self.sampler = TRGSampler(flow_step, predictor.to(self.device), T=cfg["T"],
                                  guidance_scale=cfg["guidance_scale"])
        self.actions = 0
        self.target = None

    def _weights(self, path):
        return torch.load(repo_path(path), map_location=self.device, weights_only=True)["model_state_dict"]

    def normalise(self, velocity):
        """Physical velocity field [3, Nx, Nz, Ny] -> min-max normalised fluctuations in [0, 1]."""
        fluct = velocity.astype(np.float32) - self.mean
        span = (self.maxs - self.mins).astype(np.float32)
        return ((fluct - self.mins[:, None, None, None]) / span[:, None, None, None]).clip(0.0, 1.0)

    def reward(self, velocity, first_substep):
        """Negative RMSE to the current target; refreshes the target every ``horizon`` actions."""
        x = self.normalise(velocity)
        if first_substep:
            if self.actions % self.cfg["horizon"] == 0:
                torch.set_num_threads(self.num_threads)
                x0 = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
                self.target = self.sampler.sample(x0).squeeze(0).detach().cpu().numpy()
                torch.set_num_threads(1)
            self.actions += 1
        return -float(np.sqrt(np.mean((x - self.target) ** 2)))
