# NAVSIM-trained planners on the parking dataset (zero-shot): shared adapter + TransFuser / Ego-status MLP

> **Setup (shared with all methods)**, from the repo root after `source env.sh`:
> - code: `bash setup/fetch_third_party.sh`
> - env: `bash setup/create_env.sh olp_navsim`; `methods/navsim/setup_env.sh` does the same
> - weights: `bash setup/download_ckpts.sh navsim`
> - derived inputs: top-level README, "Run a single method"
>
> Paths below are relative to the repo root. `$OLP_DERIVED` and `$CONDA_BASE` are
> set by `env.sh`.

This directory has two parts:

1. **The shared dataset → NAVSIM adapter** (`olp_navsim/`, `run_agent.py`). It is used by every
   NAVSIM-trained method in this repo: `methods/{navsim,diffusiondrive,gtrs}`.
2. **Two official NAVSIM baselines**: TransFuser and the Ego-status MLP.

Everything is inference only: no training and no fine-tuning.

| variant (`outputs/<variant>`) | method | README |
|---|---|---|
| `navsim_transfuser`, `navsim_ego_mlp` | NAVSIM baselines | this file |
| `navsim_diffusiondrive` | DiffusionDrive | `methods/diffusiondrive/README.md` |
| `navsim_hydra_mdp` | Hydra-MDP (image only, V2-99) | `methods/gtrs/README.md` |

## 1. Environment

All NAVSIM methods here share one conda env, `olp_navsim`, created by `setup/create_env.sh olp_navsim`:

- python 3.9, torch 2.0.1 (cu117 wheel), nuplan-devkit 1.2 from git, pytorch-lightning 2.2.1,
  timm 1.0.26, hydra-core 1.2;
- `diffusers==0.32.2` and `einops`, for DiffusionDrive. Newer diffusers (≥ 0.33) need torch ≥ 2.1.

The exact package list is `setup/envs/olp_navsim.txt`: the dependency closure of NAVSIM's
`requirements.txt` without the notebook stack. The code comes from `setup/fetch_third_party.sh`
and the checkpoints from `setup/download_ckpts.sh navsim diffusiondrive gtrs`.

Every method brings its own fork of the `navsim` package (NAVSIM v1.1 or v2 based). `run_agent.py`
puts the method's code dir first on `sys.path` and prints which `navsim` package it imported. No
`navsim` package is pip-installed into the env, so always go through `run_agent.py`.

| code | commit | used by |
|---|---|---|
| `third_party/navsim` (autonomousvision/navsim) | `0a380a9` (v2.2+1) | TransFuser, Ego-MLP |
| `third_party/DiffusionDrive_navsim` (hustvl/DiffusionDrive, main) | `9b52ed0` | DiffusionDrive |
| `third_party/GTRS` (NVlabs) | `92a740d` | Hydra-MDP |

The checkpoints (HF `autonomousvision/navsim_baselines`, rev `a068068`) are stored as `ckpts/navsim_baselines/{transfuser/transfuser_seed_0.ckpt, ego_status_mlp/ego_status_mlp_seed_0.ckpt}`.

**v1.1 or v2?** The baseline checkpoints were trained with PL 2.2.1 for the 2024 leaderboard (v1.x).
The TransFuser model and feature code is identical in navsim v1.1 (the copy inside DiffusionDrive)
and v2.2 (`diff -w`): same 1024×256 l0|f0|r0 crop, same LiDAR splat, same `[command, v, a]` status.
The only difference is constructor plumbing (the `trajectory_sampling` argument). Both checkpoints
load **strictly** into the v2.2 code.
They also reproduce sensible plans on real NAVSIM data (§5).

## 2. Input adaptation (dataset → NAVSIM `AgentInput`)

Everything in this section was measured on the local NAVSIM mini split (`$NAVSIM_MINI_ROOT`:
64 logs, `navsim_logs/mini` + `sensor_blobs/mini`). It is implemented in `olp_navsim/` and only uses
`olpcommon`.

### Frames

- The NAVSIM ego frame is nuPlan's rear-axle frame: x forward, y left, z up. The OpenScene infos have
  `lidar2ego = identity`, so the merged LiDAR cloud and the camera extrinsics (`sensor2lidar_*`) are in
  this frame.
- **The ground is at z ≈ −0.33 m, not at 0.** We measured this as the median of the per-cloud z mode
  over 26 clouds (p10/p90 −0.35/−0.29). The dataset ego frame has the same axes and the same origin (rear
  axle), with the ground at z = −0.45 m (`olpcommon.GROUND_Z_EGO`).
- So x and y are used unchanged. LiDAR points and camera positions are shifted by
  **z += (−0.33) − (−0.45) = +0.12 m**.
- Shifting by +0.45 m instead (i.e. ground at z = 0) would put the dataset ground 0.33 m above the nuPlan
  ground, and TransFuser's `lidar_split_height = 0.2` would then let more low clutter into its single
  "above-ground" channel.
- Output poses are NAVSIM `(x, y, heading)` of the rear axle in the ego frame at t0, at 0.5…4.0 s.
  That is exactly the olpcommon output frame, so **no conversion is applied** (`ref_traj_to_ego` is not
  used).

### Cameras (virtual nuPlan cameras, `olp_navsim/virtual_cam.py`, `make_navsim_virtual.py`)

**The nuPlan camera model.** The nominal model, used by 20624 of 20976 camera records in mini, is:

| | value |
|---|---|
| resolution | 1920×1080 |
| K | f = 1545, c = (960, 560) |
| distortion (OpenCV) | D = [−0.3561, 0.1725, −0.0021, 0.0005, −0.0523] |

**NAVSIM images are not rectified.** We projected LiDAR depth edges into 138 mini images:

| projection model | mean image gradient at the edges |
|---|---|
| K + D | 47.5 |
| pinhole K | 41.1 |

K + D wins in 90% of the images. So the models were trained on images with strong barrel distortion,
and we reproduce it. The mean mounting directions from the 2622 sampled records are:

| cam | yaw | pitch | source camera (yaw) |
|---|---|---|---|
| F0 | −0.0° | −1.4° | CAM_5 (2.9°) |
| L0 | 55.1° | −0.8° | CAM_1 (50.8°) |
| R0 | −55.1° | 0.0° | CAM_2 (−48.4°) |
| B0 | 179.9° | −0.1° | CAM_6 |
| L1 / R1 | ±111.7° | | CAM_4 / CAM_3 |
| L2 / R2 | ±141° | | CAM_4+CAM_6 / CAM_3+CAM_6 (composited) |

**Building each virtual view.** For every output pixel of a virtual view:

1. Undistort it with the nuPlan K and D (OpenCV, iterative; round-trip error < 0.05 px).
2. Rotate the ray into the ego frame with the **nuPlan mounting rotation**.
3. Project the ray into the source pinhole camera, keeping the source optical centre.
4. Sample bilinearly.

So this is the `tools/make_nusc_virtual.py` idea (match the target K and size, keep the real camera
centre). It also matches the target **lens distortion and mounting rotation**, because the NAVSIM
models use fixed crops and have no explicit camera geometry.

- Pixels outside the source image, or outside its circular vignette (r > 1350 px; the brightness drops at
  about 1400 px), are taken from the next camera in the list.
- Only L2 and R2 need this: about 30% of each comes from CAM_6. F0, L0, R0, B0, L1 and R1 are 100%
  single-camera.
- The camera centres differ from nuPlan's. The dataset front cameras are 2.01–2.05 m above ground;
  nuPlan's are 1.85 m, so the dataset cameras sit 0.17–0.20 m higher.
- The TransFuser stitch (`l0[28:-28,416:-416] | f0[28:-28] | r0[...]`) then covers yaw ±75° and pitch
  −19.7…+18.3°, as in training. Compare the top
  strip of the `viz_bev.py` figures with a real NAVSIM stitch.
- No colour or ISP matching is applied. The vehicle body visible in nuPlan L0/R0 corners is not
  painted in.
- The images are pre-rendered to
  `$OLP_DERIVED/navsim_virtual/distorted/<scene>/<CAM_X>/<token>.jpg`
  (JPEG q95, about 2 min for 960 keyframes × 8 cams with 10 workers). Missing files are rendered on
  the fly. `--no-distort` / `make_navsim_virtual.py --no-distort` gives a pinhole-K ablation.
- `Camera.sensor2lidar_rotation` = the nuPlan mount. `sensor2lidar_translation` = the source camera
  position + 0.12 m z. `intrinsics` / `distortion` = the nuPlan K / D. None of the agents here reads
  these fields.

### LiDAR (`OLPNavsimInputs.lidar_pc`)

- We use the keyframe sweep only (`ParkingDataset.lidar(tok)`, about 191k points, 128 beams), transformed
  with `sensor2ego` and then shifted by z += 0.12.
- Points inside the ego footprint (+0.25 m) are removed. nuPlan's merged cloud has about 0 points
  there; this dataset has about 250 (body and mount returns).
- The output is a (6, N) array `[x, y, z, intensity, ring, lidar_id=0]`.
- **No sweep aggregation.** The single sweep of this dataset is already denser than nuPlan's merged 5-LiDAR cloud.
  Measured on the TransFuser 256×256 / 0.25 m grid above the split height:

  | | non-empty cells | saturated cells | points |
  |---|---|---|---|
  | this dataset (one sweep) | 14% | 6.8% | 113k |
  | nuPlan | 7% | 2.3% | 41k |

  Part of this gap is the parking-lot clutter. `--lidar-sweeps N` merges N motion-compensated 10 Hz
  sweeps if wanted.

### Ego status, history and command

- **History:** 4 frames at 2 Hz (t = −1.5, −1, −0.5, 0 s). These are the previous dataset keyframes,
  which are 0.50 s apart. Only the first keyframe of the recording has no predecessor; it is padded
  with itself.
- **`ego_pose`:** (x, y, yaw) of each history frame relative to t0, from the corrected poses.
- **`ego_velocity` / `ego_acceleration`:** `ParkingDataset.ego_state(h)`, which is causal (0.6 s / 1.0 s
  fits on de-duplicated poses), in that frame's own ego frame. They are negative while reversing, as in
  nuPlan.
- **`driving_command`:** one-hot **[LEFT, STRAIGHT, RIGHT, UNKNOWN]**.
  - The order was verified on mini: for index 0, the mean future offset at 4 s is y = +2.4 m and 34%
    of frames have y > 2 m. For index 2, y = −3.0 m and 42% have y < −2 m.
  - The value comes from `ParkingDataset.command(tok)` (GT future, ±2 m at 3 s). **This is the only
    privileged input.** On the subset it gives 935 STRAIGHT, 17 LEFT and 8 RIGHT.
  - The same t0 command is used for all history frames, which none of the models read.
  - NAVSIM's own command is route-based and is issued earlier: it turns on before the lateral offset
    builds up.
  - `--command straight|unknown` replaces it with a constant, which makes the input fully causal.
- No other input is privileged. No map, boxes or GT future reach any model, and all sensors are
  ≤ t0.

## 3. Run

```bash
source env.sh
# (once per scene list) virtual cameras; ~2 min CPU for 960 keyframes
$CONDA_BASE/envs/olp_navsim/bin/python methods/navsim/make_navsim_virtual.py --cams CAM_F0 CAM_L0 CAM_R0 CAM_B0 CAM_L1 CAM_R1 CAM_L2 CAM_R2
# one agent (registered names: transfuser ego_mlp diffusiondrive hydra_mdp)
CUDA_VISIBLE_DEVICES=0 $CONDA_BASE/envs/olp_navsim/bin/python \
    methods/navsim/run_agent.py --agent transfuser --variant navsim_transfuser
# or several in sequence under one GPU lock:
GPU=0 bash methods/navsim/run_gpu.sh ego_mlp transfuser diffusiondrive hydra_mdp
$CONDA_BASE/envs/olp_navsim/bin/python tools/eval_planning.py --pred outputs/navsim_transfuser/predictions.json --horizons 1 2 3
```

- Useful flags: `--limit N` (debug), `--batch-size` (default 8), `--no-distort`, `--lidar-sweeps N`,
  `--command straight`, `--keep-ego-points`, `--fp16`.
- All runs use fp32. `--dump-aux f.npz --aux-tokens ...` saves raw model outputs (e.g. TransFuser
  boxes) for plotting.
- DiffusionDrive and Hydra-MDP register themselves in `methods/{diffusiondrive,gtrs}/agents.py`. A spec
  has: code dir, checkpoint, build function, post-processing, and optionally sensors / batch size.
- CPU also works (`CUDA_VISIBLE_DEVICES=""`): TransFuser takes about 0.25 s/frame. The outputs are
  identical except for float noise; diffusion models use different random draws on CPU.

## 4. Outputs

`outputs/<variant>/predictions.json` (olpcommon format):

- **Frames:** the keyframes of `$OLP_SCENES` (default: the 960 keyframes of `scenes_subset.txt`).
- **Times:** `t` = 0.5, 1.0, …, 4.0 s (8 poses, NAVSIM `TrajectorySampling(4 s, 0.5 s)`).
- **`trajs`:** rear-axle xy in the t0 ego frame. `trajs[0]` is the model's own `trajectory` output.
  - TransFuser and Ego-MLP are single-mode (K = 1); DiffusionDrive K = 20, Hydra-MDP K = 32.
- **`extra`:**
  - `heading`: the predicted yaw per pose
  - `command`: the command that was fed in
  - `ego_velocity` / `ego_acceleration`: the status that was fed in
- **`info`:** checkpoint, code dir, input description, runtime and peak memory.

## 5. Sanity checks

**Positive control on real NAVSIM data.** `navsim_mini_check.py` runs the same agent code on 120 scenes
of the NAVSIM mini split with NAVSIM's own `SceneLoader` and real sensors. It reports the L2 error of
`trajs[0]` against the human trajectory. A wiring error (checkpoint, features, output frame) would show
up here. Results:

| agent | L2@1s | L2@2s | L2@3s | L2@4s | ADE 0–4 s | min-of-K ADE |
|---|---|---|---|---|---|---|
| const-vel (reference) | 0.25 | 0.79 | 1.61 | 2.72 | 1.15 | – |
| Ego-MLP | 0.08 | 0.51 | 1.73 | 3.63 | 1.23 | – |
| TransFuser | 0.09 | 0.43 | 1.34 | 2.85 | 0.97 | – |
| DiffusionDrive | 0.10 | 0.43 | 1.39 | 2.97 | 1.01 | 0.78 (K=20) |
| Hydra-MDP | 0.18 | 0.42 | 0.99 | 1.96 | 0.75 | 0.44 (K=32) |

On these scenes the human driver is often stopped at lights, so 4 s errors are large for every
model, constant velocity included. All four agents are clearly better than constant velocity at
1–2 s, which confirms that checkpoints, features and output frame are wired correctly.

**Visual check (`viz_bev.py`).** On 7 frames, its figures showed:

- the stitched model input;
- the LiDAR the model sees, as a BEV;
- GT 0–4 s;
- each method's top-1 plan;
- TransFuser's own **agent detections**. These line up with the parked-car clusters in the LiDAR on
  all sides, so the LiDAR frame, axes and z-shift are right.

Forward-driving frames (`dir=FWD`) get forward plans. With cmd=RIGHT the plan bends right (scene-0057).
Most reversing frames still get forward plans. revRec is 0–10% for the sensor-based models;
DiffusionDrive and Hydra-MDP never plan backwards.

**Axis check (`axis_check.py`).** The test uses two frame sets:

- forward frames, where GT x(4 s) > 2 m (346 frames);
- turning frames, which in addition have |GT y(4 s)| > 2 m (71 frames).

The table gives the share of top-1 plans that also go forward, and the share whose lateral sign
matches the GT. A y-flip would give a sign agreement far below 50%.

| variant | forward → forward % | turn sign agreement % | median abs(y4 error), m |
|---|---|---|---|
| const_vel | 93.6 | 77.5 | 2.83 |
| navsim_diffusiondrive | 100.0 | 88.7 | 1.67 |
| navsim_ego_mlp | 100.0 | 67.6 | 2.13 |
| navsim_hydra_mdp | 100.0 | 62.0 | 2.76 |
| navsim_transfuser | 100.0 | 84.5 | 1.71 |

**`tools/eval_planning.py --horizons 1 2 3`** (960 frames). Cells are ADE/FDE in metres. The
const-vel row is the reference.

| variant | all @3s | FWD @3s | REV @3s | ADE/FDE @1s (all) | col% | revRec% |
|---|---|---|---|---|---|---|
| const_vel | 0.28/0.67 | 0.49/1.19 | 0.32/0.73 | 0.06/0.13 | 7.60 | 94.1 |
| navsim_transfuser | 0.77/2.36 | 0.67/1.89 | 1.11/3.48 | 0.09/0.19 | 11.67 | 9.9 |
| navsim_ego_mlp | 0.69/2.38 | 0.72/2.10 | 0.63/1.79 | 0.08/0.17 | 12.50 | 68.3 |

DiffusionDrive and Hydra-MDP rows are in their READMEs; all four are collected in §7.

- **Runtime and memory** (Quadro RTX 6000, shared with other jobs; batch 8, fp32, 960 frames,
  data loading included).
  - Peak memory is `torch.cuda.max_memory_allocated`; add about 0.3–0.4 GB of CUDA context.

  | model | wall time | forward per frame | peak memory |
  |---|---|---|---|
  | TransFuser | 53 s | 11.5 ms | 0.55 GB |
    | Ego-MLP | 2 s | 0.5 ms | 0.01 GB |

## 6. Known issues / caveats

- **Domain gap.** These are urban-driving planners (Las Vegas, Boston, Pittsburgh, Singapore). Training
  has no parking manoeuvres and almost no reversing.
  - From standstill they tend to plan accelerating forward motion. This is the nuPlan prior.
  - That makes their 3 s errors 2–3× const-vel on this mostly slow or stopped subset.
  - The Ego-MLP "follows" negative velocity, so its revRec is high, but it is blind.
- **Camera mismatch.** The camera heights differ by 0.17–0.20 m, the colours and ISP differ, the
  dataset cameras have a circular vignette (cropped away) and nuPlan shows its own body in the corners. Distortion,
  intrinsics and mounting yaw/pitch are matched.
  - **Ablation** (development run, output not included): TransFuser with pinhole virtual cameras
    (`--no-distort`) scores ADE/FDE@3s 0.76/2.32 (all), against 0.77/2.36 with the nuPlan
    distortion. revRec is 7.9% against 9.9%; the axis check is 100% / 83% against 100% / 85%.
  - So TransFuser is insensitive to the lens model here. The LiDAR branch dominates. The distorted
    version stays the default because it matches training.
- **LiDAR.** This is a single 128-beam sweep, not 5 merged LiDARs, so the density differs (see above).
  The intensity scale is the dataset's own, which is unused by all these models.
- **Command.** The command uses the GT future (privileged, as in the nuScenes protocol) and a ±2 m
  lateral rule at 3 s. NAVSIM's route-based command is issued earlier.
- **History.** History frames are previous keyframes (0.5 s apart, exactly like NAVSIM).
  `ego_status[-1]` is all that TransFuser and Ego-MLP use.
- **Reproducibility.** TransFuser, Ego-MLP and Hydra-MDP are deterministic. DiffusionDrive samples
  noise and is run with `--seed 0`; its noise differs between CPU and GPU.

## 7. Summary

Sanity metrics come from `tools/eval_planning.py` (all 960 frames). Cells are ADE/FDE in metres.

| variant | K | all @3s | FWD @3s | REV @3s | col% | revRec% |
|---|---|---|---|---|---|---|
| navsim_transfuser | 1 | 0.77/2.36 | 0.67/1.89 | 1.11/3.48 | 11.67 | 9.9 |
| navsim_ego_mlp | 1 | 0.69/2.38 | 0.72/2.10 | 0.63/1.79 | 12.50 | 68.3 |
| navsim_diffusiondrive | 20 | 0.83/2.65 | 0.77/2.24 | 1.13/3.55 | 10.62 | 0.0 |
| navsim_hydra_mdp | 32 | 0.76/2.31 | 0.75/2.10 | 1.10/2.89 | 9.06 | 0.0 |
| const_vel (reference) | 1 | 0.28/0.67 | 0.49/1.19 | 0.32/0.73 | 7.60 | 94.1 |
