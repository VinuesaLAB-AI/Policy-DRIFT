"""ODE samplers over the frozen CFM velocity model.

The ODE from s = 0 (noise) to s = 1 (data) is split into T segments of length ds = 1/T, each
integrated with n_euler_steps / T explicit Euler steps.

``rollout`` returns the unguided trajectory used to train the reward predictor.
``TRGSampler`` applies Terminal Reward Guidance: before every segment the state is nudged by
gamma * ds * grad R_psi (pre-placement), or after it (post-placement).
"""

import torch


class FlowStep:
    """Advances the ODE state over one segment [s, s + ds]."""

    def __init__(self, model, n_euler_steps=50, T=25):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.steps_per_segment = max(1, n_euler_steps // T)
        self.ds = 1.0 / T

    @torch.no_grad()
    def __call__(self, x_s, x0, s):
        return self.model.integrate(x0=x0, x_current=x_s, s=s, ds=self.ds, steps=self.steps_per_segment)


def s_grid(T):
    ds = 1.0 / T
    return [i * ds for i in range(T + 1)]


@torch.no_grad()
def rollout(flow_step, x0, T=25, eta=None):
    """Unguided trajectory [x_{s_0}, ..., x_{s_T}] starting from eta ~ N(0, I)."""
    x = torch.randn_like(x0) if eta is None else eta.clone()
    trajectory = [x.clone()]
    for s in s_grid(T)[:-1]:
        x = flow_step(x, x0, s)
        trajectory.append(x.clone())
    return trajectory


class TRGSampler:
    def __init__(self, flow_step, reward_predictor, T=25, guidance_scale=5.0, placement="pre"):
        if placement not in ("pre", "post"):
            raise ValueError(f"placement must be 'pre' or 'post', got '{placement}'")
        self.flow_step = flow_step
        self.reward_predictor = reward_predictor.eval()
        for p in self.reward_predictor.parameters():
            p.requires_grad_(False)
        self.T = T
        self.ds = 1.0 / T
        self.s_grid = s_grid(T)
        self.guidance_scale = guidance_scale
        self.placement = placement

    def _guidance(self, x, s):
        with torch.enable_grad():
            x_g = x.detach().requires_grad_(True)
            s_batch = torch.full((x.shape[0],), s, device=x.device, dtype=x.dtype)
            grad = torch.autograd.grad(self.reward_predictor(x_g, s_batch).sum(), x_g)[0]
        return (self.ds * self.guidance_scale * grad).detach()

    def sample(self, x0, eta=None):
        """Guided terminal state for conditioning snapshots x0 of shape [B, 3, 16, 16, 64]."""
        x = torch.randn_like(x0) if eta is None else eta.clone()
        for i in range(self.T):
            s, s_next = self.s_grid[i], self.s_grid[i + 1]
            if self.placement == "pre":
                x_in = x + self._guidance(x, s) if self.guidance_scale != 0.0 else x
                x = self.flow_step(x_in, x0, s)
            else:
                x = self.flow_step(x, x0, s)
                if self.guidance_scale != 0.0:
                    x = x + self._guidance(x, s_next)
        return x
