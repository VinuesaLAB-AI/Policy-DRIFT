"""Agent-side MPI bridge to the DNS workers (drl/dns_worker.py)."""

import os
import sys

import numpy as np
from mpi4py import MPI


class SolverIO:
    def __init__(self, cfg, config_path):
        self.cfg = cfg
        self.config_path = str(config_path)
        sim, ctrl = cfg["simulation"], cfg["control"]
        self.Nx, self.Nz = sim["Nx"], sim["Nz"]
        self.nax, self.naz = ctrl["n_agents_x"], ctrl["n_agents_z"]
        self.n_proc = sim["nprocx"] * sim["nprocz"]
        self.maxprocs = max(sim["nprocx"], sim["nprocz"])
        self.minprocs = min(sim["nprocx"], sim["nprocz"])
        self.agents = [f"control_{i}_{k}" for i in range(self.nax) for k in range(self.naz)]

    def start(self, mode, restart_file, out_dir):
        info = MPI.Info.Create()
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            info.Set("env", f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        comm = MPI.COMM_SELF.Spawn(sys.executable, maxprocs=self.n_proc, info=info,
                                   args=["-m", "drl.dns_worker", self.config_path, mode, str(restart_file), str(out_dir)])
        info.Free()
        return comm

    def stop(self, comm):
        comm.Bcast([b"END", MPI.CHARACTER], root=MPI.ROOT)
        comm.Disconnect()

    def state(self, comm):
        """Observations {agent: [v'+, u'+]} at y+ = 15 and the simulation time."""
        comm.Bcast([b"STA", MPI.CHARACTER], root=MPI.ROOT)
        zchunk = self.Nz // self.maxprocs
        planes = np.zeros((2, self.Nx, self.Nz), dtype="d")
        buf = np.zeros((self.Nx, zchunk), dtype="d")
        for i in range(2):
            for iz, rank in enumerate(range(0, self.n_proc, self.minprocs)):
                comm.Recv([buf, MPI.DOUBLE], source=rank, tag=100 + rank)
                planes[i, :, iz * zchunk:(iz + 1) * zchunk] = buf

        u_tau = self.cfg["control"]["u_tau"]
        planes[0] = planes[0] / u_tau
        planes[1] = (planes[1] - np.mean(planes[1])) / u_tau

        # Resample the planes onto the agent grid (spectral interpolation).
        tmp = np.fft.irfft(np.fft.rfft(planes, axis=1), n=self.nax, axis=1) * self.nax / self.Nx
        obs = np.fft.irfft(np.fft.rfft(tmp, axis=2), n=self.naz, axis=2) * self.naz / self.Nz
        obs = np.transpose(obs, (2, 1, 0)).reshape((self.nax * self.naz, 2), order="F")

        time = np.array(0, dtype="d")
        comm.Recv([time, MPI.DOUBLE], source=0, tag=111)
        return float(time), dict(zip(self.agents, obs))

    def actuate(self, comm, actions):
        """Map agent actions in [-1, 1] to a zero-net-mass-flux wall velocity and send it."""
        ctrl = np.array([actions[a] for a in self.agents], dtype="d").reshape(self.nax, self.naz)
        ctrl = (ctrl - np.mean(ctrl)) * self.cfg["control"]["max_amplitude"]
        ctrl = np.fft.ifft2(np.fft.fft2(ctrl)).real
        tmp = np.fft.irfft(np.fft.rfft(ctrl, axis=0), n=self.Nx, axis=0) / self.nax * self.Nx
        ctrl = np.fft.irfft(np.fft.rfft(tmp, axis=1), n=self.Nz, axis=1) / self.naz * self.Nz
        ctrl = ctrl - np.mean(ctrl)
        comm.Bcast([b"CTR", MPI.CHARACTER], root=MPI.ROOT)
        comm.Bcast([ctrl, MPI.DOUBLE], root=MPI.ROOT)

    def evolve(self, comm, steps_per_action, energy=False, tracking=False):
        """Advance one action; returns per-DNS-step wall shear (and energy / tracking terms)."""
        comm.Bcast([b"EVO", MPI.CHARACTER], root=MPI.ROOT)
        out = {}
        for key, tag, wanted in (("shear", 222, True), ("energy", 225, energy), ("tracking", 226, tracking)):
            if wanted:
                out[key] = np.zeros(steps_per_action, dtype="d")
                comm.Recv([out[key], MPI.DOUBLE], source=0, tag=tag)
        return out
