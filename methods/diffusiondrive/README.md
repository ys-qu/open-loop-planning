# DiffusionDrive (NAVSIM checkpoint), zero-shot on the parking dataset

> **Setup (shared with all methods)**, from the repo root after `source env.sh`:
> - code: `bash setup/fetch_third_party.sh`
> - env: `bash setup/create_env.sh olp_navsim`; `methods/diffusiondrive/setup_env.sh` does the same
> - weights: `bash setup/download_ckpts.sh diffusiondrive`
> - derived inputs: top-level README, "Run a single method"
>
> Paths below are relative to the repo root. `$OLP_DERIVED` and `$CONDA_BASE` are
> set by `env.sh`.

- **Paper:** B. Liao et al., "DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous
  Driving", CVPR 2025 (highlight), arXiv [2411.15139](https://arxiv.org/abs/2411.15139).
- **Code:** https://github.com/hustvl/DiffusionDrive, `main` branch (= the NAVSIM v1.1 fork), commit
  `9b52ed0`. It is cloned separately to `third_party/DiffusionDrive_navsim`.
  `third_party/DiffusionDrive` belongs to the nuScenes agent (`nusc` branch) and is not touched.
- **Checkpoint:** [`hustvl/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS`](https://huggingface.co/hustvl/DiffusionDrive)
  → `ckpts/diffusiondrive/diffusiondrive_navsim_88p1_PDMS` (88.1 PDMS on navtest).
- **Anchors:** [`kmeans_navsim_traj_20.npy`](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy)
  → `ckpts/diffusiondrive/`.
- **Loading:** all 763 checkpoint tensors load with no missing or unexpected keys (checked).

## Env

`olp_navsim` (see `methods/navsim/README.md` §1 and `setup_env.sh`). DiffusionDrive needs
`diffusers==0.32.2` and `einops`, which are already installed there. `setup_env.sh` here only calls the
shared script.

## Inputs

Same adapter and sensors as TransFuser (`methods/navsim/README.md` §2):

- virtual nuPlan cam_l0, cam_f0, cam_r0 (distorted) → the 1024×256 stitch;
- keyframe LiDAR in the NAVSIM ego frame (z + 0.12 m, ego points removed) → the 256×256 splat;
- status = [one-hot command from the GT future (privileged), v, a] from causal `ego_state`.

The agent asks for all 8 cameras, but its feature builder only reads l0, f0 and r0. The runner
therefore loads only those plus LiDAR (`sensors=` in `agents.py`).

## Inference

- `agents.py` calls the repo's `TrajectoryHead.forward_test` unchanged:
  1. Add truncated noise (t = 8) to the 20 k-means anchors.
  2. Run 2 DDIM steps.
  3. The top-1 is the argmax of the classification head.
- The method is replaced by a verbatim copy that also returns all 20 denoised trajectories and their
  logits. The top-1 is asserted to be equal.
- `--seed 0`. The noise is drawn with `torch.randn`, so CPU and GPU runs differ slightly.

## Run

```bash
cd $OLP_REPO
CUDA_VISIBLE_DEVICES=0 $CONDA_BASE/envs/olp_navsim/bin/python \
    methods/navsim/run_agent.py --agent diffusiondrive --variant navsim_diffusiondrive
python tools/eval_planning.py --pred outputs/navsim_diffusiondrive/predictions.json --horizons 1 2 3
```

## Output

`outputs/navsim_diffusiondrive/predictions.json`:

- 960 keyframes;
- `t` = 0.5…4.0 s (8 poses);
- K = 20 modes sorted by classifier logit; `trajs[0]` = the model's own top-1, and `scores` = logits;
- `extra`: `heading` (top-1 yaw), `anchor_idx_sorted`, `command`, `ego_velocity`,
  `ego_acceleration`.

## Sanity metrics (eval_planning.py, 960 frames)

Cells are ADE/FDE in metres.

| subset | n | @1s | @2s | @3s | minADE@4s (K=20) | col% | revRec% |
|---|---|---|---|---|---|---|---|
| all | 960 | 0.08/0.17 | 0.32/0.98 | 0.83/2.65 | 1.40 | 10.62 | 0.0 |
| FWD | 417 | 0.12/0.25 | 0.35/0.94 | 0.77/2.24 | 0.96 | 11.99 | – |
| REV | 101 | 0.12/0.26 | 0.45/1.33 | 1.13/3.55 | 2.11 | 15.84 | 0.0 |

**Positive control on NAVSIM mini** (120 real scenes, `methods/navsim/navsim_mini_check.py`):

- L2 at 1/2/3/4 s: 0.10 / 0.43 / 1.39 / 2.97 m
- ADE 1.01 m
- minADE (K=20) 0.78 m

## Runtime

GPU 1, Quadro RTX 6000 (shared), batch 8, fp32:

- 47 s wall time for 960 frames;
- 14.2 ms model forward per frame;
- peak allocated memory 0.57 GB.

## Known issues

- **No reversing.** None of the 20 anchors goes backwards, so revRec = 0%.
- **Accelerates from standstill.** From standstill it plans accelerating forward motion (nuPlan
  prior). It was trained on urban roads, not parking lots.
- **Other domain gaps.** Cameras, LiDAR and command differ from training: see
  `methods/navsim/README.md` §6.
