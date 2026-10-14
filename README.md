# open-loop-planning

Zero-shot open-loop planning baselines on a nuScenes-format parking-lot dataset: 9 open-source
planners (NAVSIM, nuScenes and VLA/VLM models) and 3 kinematic baselines, run on the 2 Hz
keyframes **without any training**.

## Requirements

- Linux and an NVIDIA GPU with 24 GB (Qwen2.5-VL-7B needs about 22 GB, the other models under
  10 GB), with a driver that supports CUDA 12.4. No CUDA toolkit is needed.
- conda (Miniforge, Miniconda or Anaconda).
- Disk: about 61 GB for the checkpoints, 25 GB for the four conda envs, and about 5 GB of model
  inputs per 1000 keyframes.

## Setup (once)

```bash
git clone https://github.com/ys-qu/open-loop-planning.git && cd open-loop-planning
export OLP_DATAROOT=/path/to/dataset          # unzipped dataset: the folder that contains v1.0-trainval/
source env.sh

pip install -U gdown huggingface_hub          # download tools
bash setup/fetch_third_party.sh               # 7 public repos at the tested commits
for e in olp_uniad olp_navsim olp_vlm olp_autovla; do bash setup/create_env.sh $e; done   # 5-15 min each
bash setup/download_ckpts.sh                  # all checkpoints (~61 GB)
```

You can put `export OLP_DATAROOT=...` in `env.local.sh`, which `env.sh` sources; it is not tracked by git.

## Run

```bash
source env.sh
GPU=0 bash run_recommended.sh
```

This builds the model inputs, runs the 12 methods one after another and scores them.
- Predictions go to `outputs/<method>/predictions.json`.
- A sanity-check table goes to `outputs/summary_recommended.md`.
- Each step writes a log to `logs/run_<step>.log`.
- The script ends with `all steps ok`, or lists the steps that failed.

By default it runs the 24 scenes (960 keyframes) in `scenes_subset.txt`, which takes about 4.5 h on
one RTX 6000. For other scenes, e.g. the full dataset, write a scene list (one scene name per line):

```bash
export OLP_SCENES=/path/to/scenes.txt OLP_TAG=full OLP_OUTPUTS=$OLP_REPO/outputs_full
GPU=0 bash run_recommended.sh
```

**Run a single method.** First build its inputs. `run_recommended.sh` does this itself.

```bash
$CONDA_BASE/envs/olp_uniad/bin/python tools/make_nusc_virtual.py          # nuScenes-style views (VAD, UniAD, VLMs)
$CONDA_BASE/envs/olp_navsim/bin/python methods/navsim/make_navsim_virtual.py \
    --cams CAM_F0 CAM_L0 CAM_R0 CAM_B0 CAM_L1 CAM_R1 CAM_L2 CAM_R2      # NAVSIM-style views (NAVSIM models, AutoVLA)
```

Then follow `methods/<method>/README.md`.

## Output format

```json
{"method": "...", "info": {...}, "frame": "ego",
 "predictions": {"<sample_token>": {"t": [0.5, 1.0, ...], "trajs": [[[x, y], ...], ...],
                                     "scores": [...], "extra": {...}}}}
```

- `t` is the time in seconds after the keyframe.
- `trajs` has shape K x T x 2: positions of the ego origin in the ego frame at the keyframe
  (x forward, y left, metres).
- `trajs[0]` is the method's top-1 plan.

## Methods

| Method | Trained on | Env | Horizon | K | GT command | Folder |
|---|---|---|---|---|---|---|
| Stationary, constant velocity, CTRV | - | any | 6.4 s | 1 | no | `methods/kinematic` |
| Ego-status MLP | NAVSIM | olp_navsim | 4 s | 1 | yes | `methods/navsim` |
| TransFuser | NAVSIM | olp_navsim | 4 s | 1 | yes | `methods/navsim` |
| DiffusionDrive | NAVSIM | olp_navsim | 4 s | 20 | yes | `methods/diffusiondrive` |
| Hydra-MDP (image only) | NAVSIM | olp_navsim | 4 s | 32 | yes | `methods/gtrs` |
| VAD-Base | nuScenes | olp_uniad | 3 s | 3 | yes | `methods/vad` |
| UniAD 2.0 | nuScenes | olp_uniad | 3 s | 1 | yes | `methods/uniad` |
| AutoVLA (PDMS 89) | NAVSIM | olp_autovla | 5 s | 1 | yes | `methods/autovla` |
| Impromptu-VLA 7B | nuScenes | olp_vlm | 3 s | 1 | no | `methods/impromptu_vla` |
| Qwen2.5-VL-7B, OpenEMMA-style | none | olp_vlm | 5 s | 1 | no | `methods/openemma_qwen` |

"GT command" means the method gets the driving command (left / straight / right) derived from the
GT future. This is privileged, as in the nuScenes protocol.

## Notes for evaluation

- **Same GT for every method.** `olpcommon.ParkingDataset` gives the GT used here.
  `ego_pose.json` is rotated 180° about z, so use `T_pose @ diag(-1, -1, 1)`; duplicated poses
  are removed.
- **Top-1 plan, common horizon.** Compare `trajs[0]` at a common horizon of at most 3 s; the
  horizons and K differ between methods.
- **One non-reproducible method.** Qwen2.5-VL-7B (OpenEMMA-style) is not bit-reproducible: a
  rerun changes a few frames, and ADE@3s varied by about 0.03 m. The other methods reproduce
  exactly.

Data conventions, metric definitions and expected sanity numbers are in `docs/notes.md`.
