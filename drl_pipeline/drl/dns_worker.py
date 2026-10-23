"""DNS worker spawned by the agent process (see drl/solver_io.py).

    python -m drl.dns_worker <run config> <mode> <restart file> <output dir>

``mode`` is ``train``, ``validation``, ``evaluate`` or ``generate`` (full velocity and pressure fields
every ``snapshot_every`` DNS steps). The worker idles at every action boundary
and answers three requests from the agent:

    STA  send the observation planes (v and u at y+ = 15) and the simulation time
    CTR  receive the wall actuation for the next action
    EVO  advance one action (16 DNS steps) and send the per-step reward terms
    END  stop

Reward terms sent after EVO, one value per DNS step:
    tag 222  mean wall shear du/dy|_wall
    tag 225  0.5 <|v_w|^3>_xz over the whole wall (WEN)
    tag 226  -RMSE to the TRG target (TRG, train mode)
"""

import logging
import os
import sys
import time

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
from mpi4py import MPI

from drl import config
from drl.channel import Channel

logger = logging.getLogger(__name__)
Y_PLUS_15 = -0.9166667


def add_outputs(fchannel, mode, out_dir, every, max_writes_per_file, snapshot_every):
    evaluator = fchannel.solver.evaluator
    if mode == "generate":
        state = evaluator.add_file_handler(f"{out_dir}/state", iter=snapshot_every, max_writes=np.inf, mode="overwrite")
        for name in ("u", "v", "w", "p"):
            state.add_task(name, name=name, scales=1)
        scalars = evaluator.add_file_handler(f"{out_dir}/scalars", iter=every, max_writes=np.inf, mode="overwrite")
    elif mode == "evaluate":
        planes = evaluator.add_file_handler(f"{out_dir}/planes", iter=every, max_writes=np.inf, mode="overwrite")
        planes.add_task(f"interp(v,y={Y_PLUS_15})", name="v_yp15")
        planes.add_task(f"interp(u-Ubf,y={Y_PLUS_15})", name="up_yp15")
        planes.add_task(f"interp(sqrt( ( (u-Ubf)*v )**2 ) ,y={Y_PLUS_15})", name="uv_plane")
        plane_bc = evaluator.add_file_handler(f"{out_dir}/plane_bc", iter=every, max_writes=np.inf, mode="overwrite")
        plane_bc.add_task("interp(v,y=-1)", name="v_bc")
        plane_bc.add_task("0.5*abs(interp(v,y=-1))**3 + abs(interp(v,y=-1)*interp(p,y=-1))", name="control_cost")
        profiles = evaluator.add_file_handler(f"{out_dir}/profiles", iter=every, max_writes=np.inf, mode="overwrite")
        profiles.add_task("integ(integ(u,'z'),'x')/Lx/Lz", name="umean")
        profiles.add_task("integ(integ(sqrt(v*v),'z'),'x')/Lx/Lz", name="vrms")
        profiles.add_task("integ(integ(sqrt(w*w),'z'),'x')/Lx/Lz", name="wrms")
        scalars = evaluator.add_file_handler(f"{out_dir}/scalars", iter=every, max_writes=np.inf, mode="overwrite")
    else:
        scalars = evaluator.add_file_handler(f"{out_dir}/scalars", iter=every, max_writes=max_writes_per_file,
                                             mode="append")
    scalars.add_task("integ(v*v)/Lx/Lz", name="v_en")
    scalars.add_task("integ(interp(uy,y=-1))/Lx/Lz", name="dudy_wall")
    scalars.add_task("integ( sqrt( ( (u-Ubf)*v )**2 ) )/Lx/Ly/Lz", name="uv")
    scalars.add_task("integ( sqrt( ( (u-Ubf) )**2 ) )/Lx/Ly/Lz", name="up")
    scalars.add_task("integ( sqrt( ( v**2 ) ) )/Lx/Ly/Lz", name="vp")
    scalars.add_task("integ( sqrt( ( w**2 ) ) )/Lx/Ly/Lz", name="wp")
    scalars.add_task("integ( sqrt( ( u-integ(integ(u,'z'),'x')/Lx/Lz )**2 ) )/Lx/Ly/Lz", name="up_mean")


def gather_velocity(comm, rank, nz_chunks, ny_chunks, fields, shape):
    """Assemble the full [3, Nx, Nz, Ny] velocity on rank 0 (grid layout: x local, z and y split)."""
    Nx, Nz, Ny = shape
    zc, yc = Nz // nz_chunks, Ny // ny_chunks
    if rank != 0:
        for i, f in enumerate(fields):
            comm.Send([np.array(f[:, :, :], dtype="d"), MPI.DOUBLE], dest=0, tag=(i + 1) * 1000 + rank)
        return None
    full = np.zeros((3, Nx, Nz, Ny), dtype="d")
    buf = np.zeros((Nx, zc, yc), dtype="d")
    for i in range(3):
        for iz in range(nz_chunks):
            for iy in range(ny_chunks):
                src = iz * ny_chunks + iy
                if src == 0:
                    full[i, :, iz * zc:(iz + 1) * zc, iy * yc:(iy + 1) * yc] = fields[i]
                else:
                    comm.Recv([buf, MPI.DOUBLE], source=src, tag=(i + 1) * 1000 + src)
                    full[i, :, iz * zc:(iz + 1) * zc, iy * yc:(iy + 1) * yc] = buf
    return full


def main(config_path, mode, restart_file, out_dir):
    cfg = config.load(config_path)
    sim = cfg["simulation"]
    reward_type = cfg["run"]["reward"]
    os.makedirs(out_dir, exist_ok=True)

    parent = MPI.Comm.Get_parent()
    comm = MPI.COMM_WORLD
    rank = parent.Get_rank()
    if rank != 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    Lx, Lz, Ly = sim["Lx"], sim["Lz"], sim["Ly"]
    Nx, Nz, Ny = sim["Nx"], sim["Nz"], sim["Ny"]
    dt = sim["dt"]
    steps_per_action = config.dns_steps_per_action(cfg)
    maxprocs, minprocs = max(sim["nprocx"], sim["nprocz"]), min(sim["nprocx"], sim["nprocz"])

    tracking = reward_type == "trg" and mode == "train"
    wen_energy = reward_type == "wen" and mode in ("train", "validation")
    opposition = cfg["run"].get("control") == "opposition"
    trg = None
    if tracking and rank == 0:
        from drl.trg_target import TRGTarget
        trg = TRGTarget(cfg["trg"])

    fchannel = Channel(sim, config.DATA_DIR / "turbMean.h5")
    logger.info(f"Running {mode} from {restart_file}")
    fchannel.load_restart(restart_file)

    if mode in ("train", "validation") and rank == 0:
        import re
        import shutil
        scalars_dir = os.path.join(out_dir, "scalars")
        if os.path.isdir(scalars_dir):
            for entry in os.listdir(scalars_dir):
                if re.match(r"scalars_s\d+$", entry):
                    shutil.rmtree(os.path.join(scalars_dir, entry))
    comm.Barrier()
    add_outputs(fchannel, mode, out_dir, sim["output_every"],
                int(cfg["training"]["episode_length"] * steps_per_action / sim["output_every"]),
                sim["snapshot_every"])

    flow = flow_tools.GlobalFlowProperty(fchannel.solver)
    flow.add_property("integ(integ(u,'y'),'z')/(Ly*Lz)", name="Ub")
    flow.add_property("integ( sqrt( ( (u-Ubf)*v )**2 ) )", name="uv")
    flow.add_property("p", name="p")
    divergence = (de.operators.differentiate(fchannel.u, "x") + de.operators.differentiate(fchannel.v, "y")
                  + de.operators.differentiate(fchannel.w, "z"))

    wall_shear = de.operators.interpolate(fchannel.uy, y=-1.0)
    v_wall = de.operators.interpolate(fchannel.v, y=-1.0)
    obs_planes = [de.operators.interpolate(fchannel.v, y=Y_PLUS_15), de.operators.interpolate(fchannel.u, y=Y_PLUS_15)]

    log_every = 130
    pit_time = time.time()
    while fchannel.solver.ok:
        if fchannel.solver.iteration % steps_per_action == 0:
            msg = bytearray(3)
            parent.Bcast([msg, MPI.CHARACTER], root=0)
            msg = msg.decode("utf-8")
            if msg == "END":
                break
            if msg == "STA":
                for plane in obs_planes:
                    field = plane.evaluate()
                    field.set_scales(1)
                    field = np.array(field["g"][:, :, 0], dtype="d")
                    if rank % minprocs == 0:
                        parent.Send([field, MPI.DOUBLE], dest=0, tag=100 + rank)
                if rank == 0:
                    parent.Send([np.array(fchannel.solver.sim_time, dtype="d"), MPI.DOUBLE], dest=0, tag=111)
                continue
            if msg == "CTR":
                actuation = np.zeros((Nx, Nz), dtype="d")
                parent.Bcast([actuation, MPI.DOUBLE], root=0)
                z0 = (rank // minprocs) * (Nz // maxprocs)
                ctrl_field = actuation[:, z0:z0 + Nz // maxprocs]
                continue
            if msg == "EVO":
                shear_vec = np.zeros(steps_per_action, dtype="d")
                energy_vec = np.zeros(steps_per_action, dtype="d")
                tracking_vec = np.zeros(steps_per_action, dtype="d")
                substep = 0

        if opposition:
            # Opposition control at every DNS step: v_w = -(v - <v>_xz) at y+ = 15
            plane = obs_planes[0].evaluate()
            plane.set_scales(1)
            v15 = np.array(plane["g"][:, :, 0])
            totals = np.array([v15.sum(), v15.size], dtype="d")
            comm.Allreduce(MPI.IN_PLACE, totals, op=MPI.SUM)
            ctrl_field = -(v15 - totals[0] / totals[1])

        fchannel.vbc.set_scales(1)
        fchannel.vbc["g"][:, :, 0] = ctrl_field
        fchannel.vbc["g"][:, :, :] = np.repeat(fchannel.vbc["g"][:, :, 0, np.newaxis], fchannel.vbc["g"].shape[2], axis=2)
        fchannel.v.differentiate("y", out=fchannel.vy)

        if tracking:
            fchannel.u.set_scales(1)
            fchannel.v.set_scales(1)
            fchannel.w.set_scales(1)
            velocity = [fchannel.u["g"], fchannel.v["g"], fchannel.w["g"]]
            shear_vec[substep] = de.operators.integrate(wall_shear).evaluate()["g"][0, 0, 0] / Lx / Lz / Ly
            full = gather_velocity(comm, rank, maxprocs, minprocs, velocity, (Nx, Nz, Ny))
            if rank == 0:
                tracking_vec[substep] = trg.reward(full, first_substep=(substep == 0))
            comm.Barrier()
        else:
            shear_vec[substep] = de.operators.integrate(wall_shear).evaluate()["g"][0, 0, 0] / Lx / Lz / Ly

        if wen_energy:
            vw = v_wall.evaluate()
            vw.set_scales(1)
            wall = vw["g"][:, :, 0]  # collective transform, so every rank has to request it
            local = np.abs(wall) ** 3 if rank % minprocs == 0 else np.zeros(0)
            totals = np.array([local.sum(), local.size], dtype="d")
            comm.Allreduce(MPI.IN_PLACE, totals, op=MPI.SUM)
            energy_vec[substep] = 0.5 * totals[0] / totals[1]

        if substep + 1 == steps_per_action and rank == 0:
            parent.Send([shear_vec, MPI.DOUBLE], dest=0, tag=222)
            if wen_energy:
                parent.Send([energy_vec, MPI.DOUBLE], dest=0, tag=225)
            if tracking:
                parent.Send([tracking_vec, MPI.DOUBLE], dest=0, tag=226)
        substep += 1

        fchannel.solver.step(dt)

        if fchannel.solver.iteration % log_every == 0:
            t_it = (time.time() - pit_time) / log_every
            pit_time = time.time()
            div = divergence.evaluate()
            logger.info("It: %i, t: %e, t/it: %e, max uv: %e, max div: %e" % (
                fchannel.solver.iteration, fchannel.solver.sim_time, t_it, flow.max("uv"), np.max(div["g"])))

    parent.Disconnect()


if __name__ == "__main__":
    main(*sys.argv[1:5])
