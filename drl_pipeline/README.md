# DRL pipeline

Package `drl`: Dedalus DNS of the minimal channel, the multi-agent environment and the TD3 / SAC
training and evaluation scripts.

```
drl_pipeline/
├── train.py              train a controller (Policy-DRIFT or a baseline)
├── evaluate.py           run a trained policy from the held-out initial conditions
├── generate_snapshots.py flow fields every Δt⁺ = 5 for the CFM training data
├── configs/channel.yml   simulation, control, reward, training and TRG settings
├── slurm/                SLURM templates (train.sbatch, evaluate.sbatch)
└── drl/
    ├── channel.py        Dedalus model of the open channel
    ├── dns_worker.py     DNS process group spawned by the agent
    ├── solver_io.py      MPI messages between agent and DNS workers
    ├── env.py            PettingZoo environment (256 agents, one per wall point)
    ├── trg_target.py     TRG targets and the RMSE tracking reward
    ├── metrics.py        drag reduction and actuation energy of a trajectory
    └── config.py         configuration and paths
```

## How a run works

The agent process builds the environment, which starts a group of `nprocx × nprocz` (8 × 8) DNS ranks with
`MPI_Comm_spawn` at every episode. Each action lasts 16 DNS steps (Δt⁺ = 0.6):

- every agent observes (v′⁺, u′⁺) at y⁺ = 15 above its wall point, u′ with its plane mean removed;
- the 16 × 16 actions in [−1, 1] are shifted to zero mean (zero net mass flux) and scaled by u_τ to give
  the wall-normal velocity;
- the reward is the mean over the 16 DNS steps of
  - `wse`: DR = 1 − τ_w / τ_w,0
  - `wen`: DR − ½⟨|v_w|³⟩/u_τ³
  - `trg`: −RMSE between the normalised flow field and the TRG target, which is recomputed every 8 actions
    (Δt⁺ = 5) from the current field.

Episodes last 2,500 actions (t⁺ ≈ 1,540) and start from one of the six fields in `data/restart`. Every
two episodes the deterministic policy is run from `data/validation`; the best one is saved to
`best_model/`. For `trg` runs the validation reward is DR.

## Training

```bash
python train.py --reward trg                 # Policy-DRIFT, TD3, guidance scale 5
python train.py --reward trg --gs 20         # guidance-scale sweep
python train.py --reward wse                 # TD3-WSE
python train.py --reward wen                 # TD3-WEN
python train.py --reward wse --algo sac      # SAC-WSE
python train.py --reward trg --dummy         # 2 episodes of 20 actions on a 2 × 2 process mesh
```

A full run needs 129 MPI slots: the agent, 64 training DNS ranks and 64 validation DNS ranks. On SLURM:

```bash
sbatch -A <account> -p <partition> slurm/train.sbatch --reward trg
```

Outputs in `outputs/<algo>_<reward>_seed<seed>[_gs<γ>]/`:

```
config.yml            resolved configuration of the run
checkpoints/          policy and replay buffer after every episode
best_model/           best validation policy and best_mean_reward.txt
train/, validation/   DNS scalar time series
```

A run that is restarted with the same arguments continues from its latest checkpoint.
TD3 and SAC use the Stable-Baselines3 defaults except for the settings in `configs/channel.yml`
(learning rate 10⁻⁴, batch 64, replay buffer 10⁶, 64 gradient steps every 300 actions,
critic [16, 64, 64], actor [8], TD3 exploration noise 0.2 u_τ).

## Evaluation

```bash
python evaluate.py --model ../checkpoints/drl/policy_drift_td3.zip                 # 6 trajectories, t+ ≈ 3000
python evaluate.py --model ../checkpoints/drl/policy_drift_td3.zip --inits 1 --n-outputs 130
sbatch -A <account> -p <partition> slurm/evaluate.sbatch --model ../checkpoints/drl/td3_wse.zip
```

Each trajectory is written to `outputs/evaluation/<model>/init_<i>/` (`scalars`, `plane_bc`, `planes`,
`profiles`) and summarised in `summary.json`; `scripts/table1.py` turns these into Table 1. One
trajectory takes about 20 minutes on 64 cores.

## Snapshot generation

```bash
python generate_snapshots.py --control none --out ../data/cfm/uncontrolled
python generate_snapshots.py --control opposition --out ../data/cfm/opposition_control
python generate_snapshots.py --control ../checkpoints/drl/td3_wse.zip --out ../data/cfm/td3_wse
```

Each run starts from `data/generation/snapshot_restart.h5` and writes `--n-snapshots` (default 7,350) flow
fields, one every 130 DNS steps (Δt⁺ = 5), in the format read by `flow_matching_trg/trg/data.py`. The wall
actuation is zero, opposition control (v_w = −(v − ⟨v⟩) at y⁺ = 15, updated every DNS step) or the actions
of a trained policy (updated every action).

## Reproducibility

Simulations are bitwise reproducible: two runs with the same inputs on the same 8 × 8 process mesh give
identical outputs. This relies on deterministic FFTW plans, which `drl/__init__.py` selects before
Dedalus is imported (Dedalus otherwise times several FFT algorithms at start-up and may pick different
ones in different runs, which changes results at round-off level).
