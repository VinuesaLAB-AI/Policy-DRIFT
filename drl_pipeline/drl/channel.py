"""Dedalus (v2) model of the minimal open channel.

Domain x in [0, Lx), z in [-Lz/2, Lz/2) (Fourier) and y in [-Ly, 0] (Chebyshev). The lower wall
y = -Ly is no-slip with wall-normal velocity vbc (the actuation); the top boundary y = 0 is a
free-slip, impermeable plane. The bulk velocity is held at 2/3 by the pressure gradient Fdp.
"""

import logging

import h5py
import numpy as np
from dedalus import public as de

logger = logging.getLogger(__name__)


class Channel:
    def __init__(self, sim, mean_profile_file):
        self.Lx, self.Lz, self.Ly = sim["Lx"], sim["Lz"], sim["Ly"]
        x_basis = de.Fourier("x", sim["Nx"], interval=(0, self.Lx), dealias=3 / 2)
        z_basis = de.Fourier("z", sim["Nz"], interval=(-self.Lz / 2, self.Lz / 2), dealias=3 / 2)
        y_basis = de.Chebyshev("y", sim["Ny"], interval=(-self.Ly, 0.0), dealias=3 / 2)
        self.domain = de.Domain([x_basis, z_basis, y_basis], grid_dtype=np.float64,
                                mesh=(sim["nprocx"], sim["nprocz"]))

        problem = de.IVP(self.domain, variables=["p", "u", "w", "v", "uy", "wy", "vy", "Fdp"], time="t")
        problem.parameters["Re"] = sim["Re"]
        problem.substitutions["Lap(A, Ay)"] = "dx(dx(A))+dz(dz(A))+dy(Ay)"
        problem.substitutions["Adv(A, Ay)"] = "u*dx(A)+w*dz(A)+v*Ay"
        problem.substitutions["Ub"] = str(2 / 3)
        problem.substitutions["Ly"] = str(self.Ly)
        problem.substitutions["Lx"] = str(self.Lx)
        problem.substitutions["Lz"] = str(self.Lz)

        self.vbc = self.domain.new_field(name="vbc")
        self.vbc["g"] = 0.0
        problem.parameters["vbc"] = self.vbc
        self.Ubf = self._load_mean_profile(mean_profile_file)
        problem.parameters["Ubf"] = self.Ubf

        problem.add_equation("dt(u) + dx(p) - 1/Re * Lap(u, uy) + Fdp =- Adv(u, uy)")
        problem.add_equation("dt(v) + dy(p) - 1/Re * Lap(v, vy) = - Adv(v, vy)")
        problem.add_equation("dt(w) + dz(p) - 1/Re * Lap(w, wy) = - Adv(w, wy)")
        problem.add_equation("dx(u) + vy + dz(w) = 0")
        problem.add_equation("Fdp = 0", condition="(nx != 0) or (nz != 0)")
        problem.add_equation("dy(Fdp) = 0", condition="(nx == 0) and (nz == 0)")
        problem.add_equation("integ(integ(u,'y'),'z') = Ub*Ly*Lz", condition="(nx == 0) and (nz == 0)")
        problem.add_equation("uy - dy(u) = 0")
        problem.add_equation("vy - dy(v) = 0")
        problem.add_equation("wy - dy(w) = 0")
        problem.add_bc("left(u)  = 0")
        problem.add_bc("right(uy) = 0")
        problem.add_bc("left(w)  = 0")
        problem.add_bc("right(wy) = 0")
        problem.add_bc("left(v)  = vbc(y=-0.5)", condition="(nx!=0) or (nz!=0)")
        problem.add_bc("right(v) = 0")
        problem.add_bc("right(p)  = 0", condition="(nx == 0) and (nz == 0)")

        self.problem = problem
        self.solver = problem.build_solver(de.timesteppers.RK443)
        state = self.solver.state
        self.u, self.uy, self.v, self.vy = state["u"], state["uy"], state["v"], state["vy"]
        self.w, self.wy, self.p = state["w"], state["wy"], state["p"]

    def _load_mean_profile(self, path):
        """Mean streamwise velocity Ubf (used only for diagnostics of u - Ubf)."""
        field = self.domain.new_field(name="Ubf")
        with h5py.File(path, mode="r") as f:
            dset = f["tasks"]["u"]
            for layout in self.domain.dist.layouts:
                if np.allclose(layout.grid_space, dset.attrs["grid_space"]):
                    break
            else:
                raise ValueError("No matching layout")
            scales = dset.shape[1:] / layout.global_shape(scales=1)
            scales[~layout.grid_space] = 1
            local = dset[(-1,) + layout.slices(tuple(scales))]
            field.set_scales(scales, keep_data=False)
            field[layout][tuple(slice(n) for n in local.shape)] = local
            field.set_scales(self.domain.dealias, keep_data=True)
        return field

    def load_restart(self, path):
        """Load a Dedalus restart file and restart the iteration counter at zero."""
        self.solver.load_state(path, 0)
        self.solver.iteration = 0
        self.solver.start_time = self.solver.sim_time
        self.solver.stop_sim_time = np.inf
        self.solver.stop_wall_time = np.inf
        self.solver.stop_iteration = np.inf
