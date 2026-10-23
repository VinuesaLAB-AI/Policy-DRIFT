# Checkpoints

The models behind the results in the paper. The `.pt` files are stored with Git LFS
(`git lfs install && git lfs pull` after cloning).

| File | Model | Selected at | Used for |
|---|---|---|---|
| `cfm/velocity_model.pt` | CFM velocity model v_θ (3D U-Net, 23.36M parameters) | epoch 153, lowest validation loss (0.00848) | TRG targets |
| `trg/reward_predictor.pt` | Reward predictor R_ψ (1.29M parameters), terminal reward DR − E with α = 1 | epoch 338, lowest training prediction error (0.0538) | TRG targets |
| `drl/policy_drift_td3.zip` | Policy-DRIFT, TD3 tracking TRG targets with guidance scale γ = 5 | episode 16 | Table 1, γ = 5 row of the guidance-scale sweep |
| `drl/td3_wse.zip` | TD3 trained on DR (TD3-WSE) | episode 56 | Table 1 |
| `drl/td3_wen.zip` | TD3 trained on DR − E_act (TD3-WEN) | episode 10 | Table 1 |

The DRL checkpoints are the policies with the highest mean episode reward on the validation initial
condition (`data/validation`); they are Stable-Baselines3 2.7.1 archives.
The `.pt` files hold `model_state_dict` and the selection metadata in the table:

```python
import torch
from trg.velocity_model import VelocityModel
from trg.reward_predictor import RewardPredictor

v = VelocityModel()
v.load_state_dict(torch.load("checkpoints/cfm/velocity_model.pt", weights_only=True)["model_state_dict"])
r = RewardPredictor()
r.load_state_dict(torch.load("checkpoints/trg/reward_predictor.pt", weights_only=True)["model_state_dict"])
```

## Checksums

SHA-256:

```
081cd4c92c642d5c10d8fa78cd6935f5044ee9e535f572aea054499d639456d7  cfm/velocity_model.pt
decaca489ce2dd485d85f01f572e0b644f968c2f295be97516c82242a65404e7  trg/reward_predictor.pt
7699d527f5c82fc1ef648ef8caf9e05410fe5d3036f210a08e905ca9370b7175  drl/policy_drift_td3.zip
8a57f68ba4d93705176c950c2a9c77c922fe0fd76dd74c9810fd1ae11fb5aece  drl/td3_wse.zip
5bbc0cf87be62708c663c90be85b3b7d5a3143b73f924bc0f5c58b7c45953967  drl/td3_wen.zip
```
