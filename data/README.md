# Data

All fields are from DNS of the minimal open channel at Re_τ = 180
(L_x × L_y × L_z = 2.67h × h × 0.8h, N_x × N_z × N_y = 16 × 16 × 64), non-dimensionalised with h and the
laminar centreline velocity.

| Path | Content | Used by |
|---|---|---|
| `restart/snapshots_s{1..6}.h5` | Six turbulent fields; every training episode starts from one of them, drawn with a generator seeded by `--seed` | `drl_pipeline/train.py` |
| `validation/snapshot_validation.h5` | Held-out field on which checkpoints are scored during training | `drl_pipeline/train.py` |
| `eval/snapshots_s{1..6}.h5` | The six held-out initial conditions of Table 1 | `drl_pipeline/evaluate.py` |
| `turbMean.h5` | Mean streamwise velocity profile U(y), used for the u − U diagnostics | DNS workers |
| `generation/snapshot_restart.h5` | Turbulent field from which every snapshot-generation run starts | `drl_pipeline/generate_snapshots.py` |
| `means.h5` | Per-voxel mean velocity and per-channel min / max / std of the fluctuations of the CFM training data | CFM data loader, TRG targets |
| `sample/{uncontrolled,opposition_control,td3_wse}/` | 24 consecutive snapshots (Δt⁺ = 5) of each regime | `--dummy` runs of `flow_matching_trg/` |

The restart files are Dedalus 2 state files (`tasks/{u,v,w,p,uy,vy,wy,Fdp}` on the 16 × 16 × 64 grid) and are
loaded with `solver.load_state`. Snapshot files contain `u`, `v`, `w`, `p` on the 16 × 16 × 64 grid,
the grid coordinates `x`, `y`, `z` and the `time` since the start of the run.

## CFM training data

Training the CFM model and the reward predictor needs 7,350 consecutive snapshots at Δt⁺ = 5 for each of
the uncontrolled, opposition-controlled and TD3-WSE-controlled regimes (22,050 files, about 6 GB). They are
generated with the DNS of `drl_pipeline/`, starting from `generation/snapshot_restart.h5`:

```bash
cd drl_pipeline
python generate_snapshots.py --control none --out ../data/cfm/uncontrolled
python generate_snapshots.py --control opposition --out ../data/cfm/opposition_control
python generate_snapshots.py --control ../checkpoints/drl/td3_wse.zip --out ../data/cfm/td3_wse
```

Each run needs 65 MPI slots. Opposition control sets the wall velocity to −(v − ⟨v⟩) at y⁺ = 15 at every DNS step; the
TD3-WSE regime uses the trained TD3-WSE policy. The paths are those of `flow_matching_trg/configs/dataset.yaml`.
The first 80% of every regime (in time) is used for training and the rest for testing.
`flow_matching_trg/compute_stats.py` recomputes `means.h5` from the generated data.
