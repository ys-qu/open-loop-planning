# Hydra-MDP (image-only, V2-99), zero-shot on the parking dataset

> **Setup (shared with all methods)**, from the repo root after `source env.sh`:
> - code: `bash setup/fetch_third_party.sh`
> - env: `bash setup/create_env.sh olp_navsim`; `methods/gtrs/setup_env.sh` does the same
> - weights: `bash setup/download_ckpts.sh gtrs`
> - derived inputs: top-level README, "Run a single method"
>
> Paths below are relative to the repo root. `$OLP_DERIVED` and `$CONDA_BASE` are set by
> `env.sh`.

- **Paper:** Hydra-MDP, arXiv [2406.06978](https://arxiv.org/abs/2406.06978). The released
  image-only V2-99 checkpoint comes with the GTRS code base: Z. Li et al., "Generalized
  Trajectory Scoring for End-to-end Multimodal Planning", arXiv
  [2506.06664](https://arxiv.org/abs/2506.06664).
- **Code:** https://github.com/NVlabs/GTRS @ `92a740d`, cloned to `third_party/GTRS` (a NAVSIM v2
  fork). It ships the 16384-trajectory vocabulary in `traj_final/16384.npy`.
- **Checkpoints:** HF [`Zzxxxxxxxx/gtrs`](https://huggingface.co/Zzxxxxxxxx/gtrs), saved to `ckpts/gtrs/`.
  - `hydra_mdp_vov.ckpt`: Hydra-MDP, image only, 37.5 EPDMS on navhard.
  - `dd3d_det_final.pth`: the V2-99 DD3D initialisation. The model constructor insists on loading
    it, and the full checkpoint then overwrites it.
- **Loading:** `strict=False`. The only missing key is `model._trajectory_head.vocab`, which the repo
  deliberately loads from the npy.

## Env

`olp_navsim` (`methods/navsim/README.md` §1).

- `xformers` from GTRS' requirements is **not** needed. Only the DINOv2 layers use it; the V2-99 path
  uses the pure-PyTorch chunked attention in `navsim/agents/utils/attn.py`.
- Runs in fp32; neither FlashAttention nor bf16 is used.

## Model and inputs

Hydra-MDP is a trajectory-scoring planner:

- A **V2-99** image backbone runs on the l0|f0|r0 stitch at **2048×512**. That is the TransFuser
  crop at twice the resolution, giving 16×64 tokens.
- A transformer scores **16384 vocabulary trajectories** (4 s at 10 Hz, rear-axle x/y/yaw).
  - Its heads imitate the PDM sub-scores (NC, DAC, TTC, EP, DDC, LK, TL) plus an imitation head.
  - They are conditioned on the ego status `[command, v, a]`.
- **Top-1** = argmax of the repo's hard-coded inference score in `HydraTrajHead.forward`.
- **No LiDAR.**
- The feature builder also stitches the l2|b0|r2 back view and keeps 2 history frames
  (seq_len = 2). The model only uses the current front stitch, but the inputs are provided as in
  training.
- Virtual cameras: see `methods/navsim/README.md` §2. B0/L2/R2 are rendered too; L2 and R2 are
  composites of CAM_4/CAM_3 + CAM_6.
- **Command:** GT-derived one-hot (privileged), as for all NAVSIM agents.

The implementation is in `agents.py`. The config is copied from
`navsim/planning/script/config/common/agent/hydra_mdp_vov.yaml`.

## Run

```bash
source env.sh
CUDA_VISIBLE_DEVICES=0 $CONDA_BASE/envs/olp_navsim/bin/python \
    methods/navsim/run_agent.py --agent hydra_mdp --variant navsim_hydra_mdp
$CONDA_BASE/envs/olp_navsim/bin/python tools/eval_planning.py --pred outputs/navsim_hydra_mdp/predictions.json --horizons 1 2 3
```

## Output

`outputs/navsim_hydra_mdp/predictions.json`:

- The 10 Hz vocabulary trajectories are sub-sampled to 2 Hz: `t` = 0.5…4.0 s.
- K = 32: the 32 best vocabulary entries by the inference score (`scores`, log-domain).
  `trajs[0]` = the model's own `trajectory` (asserted).
- `extra`: `heading`, `vocab_idx_top` (first 8 vocabulary indices), `command`, `ego_velocity`,
  `ego_acceleration`.

## Sanity metrics (eval_planning.py, 960 frames)

Cells are ADE/FDE in metres.

| subset | @1s | @2s | @3s | minADE@4s (K=32) | col% | revRec% |
|---|---|---|---|---|---|---|
| all | 0.17/0.29 | 0.35/0.81 | 0.76/2.31 | 1.02 | 9.06 | 0.0 |
| FWD | 0.21/0.36 | 0.39/0.82 | 0.75/2.10 | 0.90 | 9.35 | – |
| REV | 0.31/0.55 | 0.60/1.25 | 1.10/2.89 | 1.59 | 12.87 | 0.0 |

- **1 s errors are higher than for regression models.** The plan is a discrete vocabulary
  entry, so it does not match the current speed exactly.
- **Hydra-MDP never picks a backward vocabulary entry**, even on reversing frames.

## Positive control (NAVSIM mini, `methods/navsim/navsim_mini_check.py`)

On 120 real NAVSIM mini scenes, Hydra-MDP is the most accurate of the NAVSIM agents here. For
reference, const-vel scores ADE 1.15 and TransFuser 0.97 on the same scenes.

| L2@1s | L2@2s | L2@3s | L2@4s | ADE | minADE (K=32) | long. offset @4s |
|---|---|---|---|---|---|---|
| 0.18 | 0.42 | 0.99 | 1.96 | 0.75 | 0.44 | +0.79 m |

## Runtime

On a Quadro RTX 6000 shared with other jobs (batch 8, fp32):
- about 350 s wall time for 960 frames;
- 0.33 s forward per frame (V2-99 at 2048×512 plus the 16384-trajectory transformer);
- peak allocated memory 4.9 GB.

## Known issues

- **Fixed vocabulary.** The 4 s trajectories come from navtrain, so there are no parking
  manoeuvres and almost no reversing.
- **Training data mismatch.** The PDM-imitating heads were trained with nuPlan maps. In a parking
  lot, "drivable area" and "lane keeping" have no meaning comparable to nuPlan's.
- **Unused inputs.** The back-view inputs (L2/R2 composites) are computed but this checkpoint does
  not use them (`use_back_view=False`).
