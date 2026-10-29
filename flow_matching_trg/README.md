# Conditional flow matching and Terminal Reward Guidance

Package `trg` and the scripts that train the CFM velocity model and the TRG reward predictor.

```
flow_matching_trg/
├── train_cfm.py          CFM velocity model training
├── train_trg.py          reward predictor training on unguided CFM rollouts
├── compute_stats.py      normalisation statistics (data/means.h5)
├── configs/
│   ├── dataset.yaml      snapshot directories, normalisation and reward constants
│   ├── cfm.yaml          CFM hyperparameters
│   ├── trg.yaml          reward predictor hyperparameters
│   └── dummy.yaml        overrides for --dummy runs on data/sample
└── trg/
    ├── unet.py           3D U-Net backbone
    ├── velocity_model.py v_θ(u_s, s, u0), CFM loss and Euler integration
    ├── reward_predictor.py  R_ψ(u_s, s)
    ├── reward.py         terminal reward R = DR − α E on normalised fields
    ├── sampler.py        unguided rollouts and the TRG-guided sampler
    ├── data.py           snapshot-pair dataset
    └── config.py         configuration loading
```

## Method

Fields are fluctuations about the per-voxel mean, min-max normalised per channel to [0, 1]
(`data/means.h5`, `configs/dataset.yaml`).

**CFM.** With u_s = (1 − s) η + s u₁ and η ~ N(0, I), the U-Net v_θ(u_s, s, u₀) is trained on
‖v_θ − (u₁ − η)‖². Sampling integrates du/ds = v_θ from s = 0 to 1 with 50 explicit Euler steps.

**Reward predictor.** The ODE is split into T = 25 segments of two Euler steps. For each batch of
conditioning snapshots an unguided rollout gives u_{s_0}, …, u_{s_T}; R_ψ(u_{s_i}, s_i) is regressed onto
the terminal reward r_T = R(u_{s_T}) with weight s_i. R = DR − α E with α = 1, where DR and E are the drag
reduction and the actuation energy computed on the normalised field (see `trg/reward.py`).

**TRG.** Before every segment the state is nudged by γ · (1/T) · ∇R_ψ (pre-placement) and then advanced by
the frozen v_θ. The paper uses γ = 5.

## Usage

```bash
python train_cfm.py --dummy      # 2 epochs on data/sample, a few minutes on a CPU
python train_trg.py --dummy

python train_cfm.py              # paper configuration, needs data/cfm (see data/README.md)
python train_trg.py cfm_checkpoint=flow_matching_trg/outputs/cfm/best.pt
```

Any configuration entry can be overridden as `key=value`, e.g. `training.batch_size=64`. Outputs
(`best.pt`, `latest.pt`, TensorBoard logs) go to `outputs/cfm` and `outputs/trg`. Training runs on
one GPU when available.

Settings (`configs/cfm.yaml`, `configs/trg.yaml`):

| | CFM velocity model | Reward predictor |
|---|---|---|
| Optimiser | AdamW, lr 10⁻³, weight decay 10⁻² | AdamW, lr 10⁻⁴, weight decay 10⁻² |
| Schedule | ReduceLROnPlateau on the validation loss (patience 20, factor 0.5) | cosine annealing over 500 epochs |
| Batch size | 128 | 24 |
| Gradient clipping | 1.0 | 1.0 |
| Epochs | up to 500, early stopping after 50 epochs without improvement | up to 500, early stopping after 50 epochs without improvement |
| Checkpoint | lowest validation loss | lowest training prediction error at the terminal state |
