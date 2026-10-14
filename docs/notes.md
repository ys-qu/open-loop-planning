# Notes: data, shared code, metrics, evaluation

Background for writing a new adapter or running the formal evaluation. The README only covers
how to install and run.

## 1. Data

| What | Where |
|---|---|
| Unzipped dataset (`--dataroot`) | `$OLP_DATAROOT` |
| Derived data (virtual camera renders, infos) | `$OLP_DERIVED/` (default `./derived`) |
| Scene list with sensor data in the shipped subset (24 scenes, 960 keyframes) | `scenes_subset.txt` |

### Conventions and data issues (read before writing an adapter)

1. **Ego frame:** x forward, y left, z up. The origin is about at the rear axle; the
   ground is at z ≈ −0.45 m (`olpcommon.GROUND_Z_EGO`). The LiDAR is about 2.18 m above
   ground. `calibrated_sensor.json` extrinsics and the agent boxes in
   `labels/agents_bev_lc_refined_human_corr` are in this frame. The LiDAR frame ≈ the ego
   frame, with an offset of (0.663, 0, 1.733). The images are well described by the given
   pinhole K (checked by LiDAR projection) despite the circular vignette.
2. **ego_pose.json is rotated 180° about z.** Its +x axis points to the rear of the car.
   Use `T_world_from_ego = T_pose @ diag(-1, -1, 1)`.
   - Evidence: consecutive LiDAR sweeps align with a median NN distance of 0.16 m with
     the fix, versus 0.75 m without it.
   - Consequence: `labels/ego_trajectory.json` `*_traj_rel` inherit the flip. Forward
     motion has negative x there.
3. **ego_pose.json is nominally 10 Hz but changes only about every 200 ms.** 41% of
   consecutive entries are exact duplicates.
   - `ParkingDataset(pose_mode='dedup')` (the default) keeps the first entry of each run of
     duplicates, shifted by −50 ms, then interpolates.
   - Against LiDAR ICP odometry, the mean 0.1 s step error is 0.075 m with dedup versus
     0.185 m for the raw table. `pose_mode='raw'` reproduces the raw table.
4. **Camera mapping** (by mounting yaw): CAM_5 = FRONT, CAM_1 = FRONT_LEFT,
   CAM_2 = FRONT_RIGHT, CAM_4 = BACK_LEFT, CAM_3 = BACK_RIGHT, CAM_6 = BACK.
   - All cameras are 2880×1860 with f ≈ 1200 px (HFOV ≈ 100°).
   - Camera sweeps are about 7.5 Hz; some 10 Hz `sample_data` rows point to the same file.
5. `parking_lots/.../parking_spots.json` in this release has only 8 stalls, while
   `spot_occupancy.json` references 463. The full stall map is missing.
6. The `sample_annotation` / `instance` tables are empty. Agent boxes live in
   `labels/agents_bev_lc_refined_human_corr/<scene>_<token>.json`:
   - **Frame:** the ego frame of the box's own keyframe. A 2D fit against LiDAR gives the
     best match at 0 m offset.
   - **Box fields:** `bev_size` = [length, width]; `bev_heading` is measured from +x;
     `bev_z` is the box **bottom**. For cars its median is −0.44 m, i.e. the ground.
   - **Duplicate keyframes:** each scene file also holds 4 pre-context and 12
     post-context keyframes that belong to the neighbouring scenes. For those keyframes,
     use the copy in the **owning scene's file**. The context copies are partial, e.g. 23
     instead of 66 boxes. `ParkingDataset` does this.
7. **Pose jumps while standing still.** `ego_pose.json` occasionally jumps by about
   0.24 m while the car is stationary; the raw table has this too. On 6 of the 960
   keyframes, `ego_state()` therefore reports a spurious |v| > 0.2 m/s: the first keyframe
   of scene-0057, plus one each in scenes 0063, 0064, 0066, 0069 and 0077. This feeds
   every ego-status input.
8. **`prev`/`next` cross scene boundaries.** Both `ParkingDataset.prev_keyframe()` and the
   virtual infos link to the neighbouring scene's keyframe, because the recording is
   continuous. Temporal models must still **reset on a `scene_token` change**, as in the
   nuScenes protocol; all adapters here do.
9. **The command is almost always STRAIGHT.** `ParkingDataset.command()` gives STRAIGHT on 935
   of 960 keyframes: lateral offset of the rear axle at 3 s, ±2 m threshold. The nuScenes
   converters apply the same rule to the LiDAR position (0.94 m further ahead); that would
   differ on 27 of the 960 frames.
10. **Leakage rule:** inputs may use only data up to the keyframe. The one exception is the
   high-level driving command: nuScenes and NAVSIM models need it, and
   `ParkingDataset.command()` derives it from the GT future, exactly like the nuScenes
   protocol. Every method README states which inputs are privileged.

## 2. Shared code

- `olpcommon/` contains only numpy, so it imports from any env (python ≥ 3.7).
  - `ParkingDataset(dataroot, scenes)`:
    - `.keyframes`, `.camera(tok, 'CAM_FRONT' | 'CAM_5')`, `.cameras(tok)`, `.lidar(tok)`
    - `.sweeps(tok, ch, n)`
    - `.ego2global(tok)` (corrected), `.ego2global_at(ts)`
    - `.history(tok, n, dt)`, `.ego_state(tok)` (causal velocity, acceleration, yaw rate)
    - `.command(tok)`, `.gt_future(tok, horizon, dt)`, `.agent_boxes(tok, ref_token)`
    - `.direction(tok)` (FWD / STP / REV)
  - `olpcommon.geometry.ref_traj_to_ego()` converts a trajectory of a vehicle-fixed point
    (for example the nuScenes LiDAR) to the ego origin.
  - `olpcommon.predictions` holds the common output format (below).
- `tools/make_nusc_virtual.py` builds a virtual nuScenes view for nuScenes-trained models:
  - Each camera is rescaled to the focal length of the matching nuScenes camera and
    cropped to 1600×900.
  - It defines a virtual nuScenes LIDAR_TOP frame (x right, y forward, 1.84 m above
    ground, 0.94 m ahead of the rear axle).
  - It writes the can_bus fields from the poses and a nuScenes-style command.
  - Output: `$OLP_DERIVED/nusc_virtual/{samples/,infos_$OLP_TAG.json}` (`OLP_TAG` defaults to `subset`).
- `tools/eval_planning.py` is the sanity-check metric script (section 4).
- `tools/summarize.py` turns its json into a markdown table.
- `tools/visualize.py` draws the front camera plus a BEV of the GT, labelled agents,
  several methods' plans and the first method's own detections and map. It takes
  `--tokens auto|<prefixes>` and writes to `outputs/vis/` by default.

## 3. Prediction format

Each method writes `outputs/<method>/predictions.json`:

```json
{"method": "...", "info": {...}, "frame": "ego",
 "predictions": {"<sample_token>": {"t": [0.5, 1.0, ...],
                                     "trajs": [[[x, y], ...], ...],
                                     "scores": [...], "extra": {...}}}}
```

- `t` is seconds after the keyframe.
- `trajs` has shape K×T×2 and holds positions of the ego origin in the ego frame at the
  keyframe (x forward, y left, metres). `trajs[0]` is the method's own top-1 plan.
- `extra` holds anything method-specific, such as VLM text or the command that was used.

## 4. Sanity-check metrics

```bash
$CONDA_BASE/envs/olp_uniad/bin/python tools/eval_planning.py --pred outputs/*/predictions.json --horizons 1 2 3 --out outputs/summary.json
```

It reports:
- Top-1 ADE/FDE@H and minADE/minFDE@H, both on a 10 Hz grid (predictions are linearly
  interpolated).
- `col%`: the top-1 ego footprint overlaps a labeled agent box at a future keyframe within
  3 s.
- `colGT%`: the same check run on the GT trajectory, i.e. the label-noise floor.
  - It is 4.1%, from pedestrians passing within about 1 m and from a large truck parked
    right next to the ego's path.
  - The ego footprint is assumed to be 4.94 × 2.0 m with its centre 1.4 m ahead of the
    ego origin (`olpcommon.EGO_*`).
  - Headings along a plan come from `headings_from_xy`, which limits turning to a 4.5 m
    radius.
- `revRec%`: on frames where the ego reverses, the share whose top-1 plan goes backwards.
- Breakdowns by ego direction (FWD / STP / REV) and by distance travelled in 6.4 s.

## 5. GPUs

Everything was developed on two Quadro RTX 6000s (Turing sm_75, 24 GB), so no bf16 tensor
cores and no FlashAttention-2. The code therefore uses fp32/fp16 and standard attention,
which also runs on newer GPUs.

The run scripts take `GPU=<id>` and wrap each job in `flock locks/gpu<id>.lock`, so two
jobs never share a card unintentionally. Peak memory is under 10 GB for most planners and
about 21 GB for Qwen2.5-VL-7B.

## 6. Expected sanity numbers

What our run gave on the 960 subset keyframes (`scenes_subset.txt`), as a check for a new
install. These are **sanity checks only**: `tools/eval_planning.py`, top-1 plan, 10 Hz grid,
3 s horizon. `run_recommended.sh` writes the full table, with per-direction and
per-travel-distance rows, to `outputs/summary_recommended.md`.

Column meanings:
- **cmd?**: the method consumes the GT-derived high-level command (privileged; see §1.10).
- **ego?**: the method consumes ego state or past motion (causal, from the poses).
- **H**: the method's own planning horizon in seconds.
- **K**: the number of trajectories per frame.
- **col%**: collision rate within 3 s. The GT trajectory itself scores 4.1% (label-noise floor).
- **revRec%**: on frames where the ego reverses, the share whose plan also reverses.

| Method | Family / train data | Env | cmd? | ego? | H | K | ADE/FDE@3s | col% | revRec% | Doc |
|---|---|---|---|---|---|---|---|---|---|---|
| stationary | kinematic | any | – | – | 6.4 | 1 | 0.94 / 1.82 | 4.2 | 0 | `methods/kinematic` |
| constant velocity | kinematic | any | – | yes | 6.4 | 1 | 0.28 / 0.67 | 7.6 | 94 | `methods/kinematic` |
| CTRV | kinematic | any | – | yes | 6.4 | 1 | 0.25 / 0.60 | 6.4 | 94 | `methods/kinematic` |
| Ego-status MLP | NAVSIM | olp_navsim | yes | yes | 4 | 1 | 0.69 / 2.38 | 12.5 | 68 | `methods/navsim` (blind: no sensors) |
| TransFuser | NAVSIM | olp_navsim | yes | yes | 4 | 1 | 0.77 / 2.36 | 11.7 | 10 | `methods/navsim` |
| DiffusionDrive | NAVSIM | olp_navsim | yes | yes | 4 | 20 | 0.83 / 2.65 | 10.6 | 0 | `methods/diffusiondrive` |
| Hydra-MDP (image only) | NAVSIM | olp_navsim | yes | yes | 4 | 32 | 0.76 / 2.31 | 9.1 | 0 | `methods/gtrs` |
| VAD-Base | nuScenes E2E | olp_uniad | yes | can_bus | 3 | 3 | 0.72 / 1.37 | 6.2 | 0 | `methods/vad` |
| UniAD 2.0 | nuScenes E2E | olp_uniad | yes | can_bus | 3 | 1 | 0.85 / 1.68 | 19.3 | 0 | `methods/uniad` |
| AutoVLA (PDMS 89) | VLA, NAVSIM | olp_autovla | yes | yes | 5 | 1 | 0.93 / 2.90 | 12.2 | 0 | `methods/autovla` |
| Impromptu-VLA 7B | VLA, nuScenes ft | olp_vlm | – | yes | 3 | 1 | 0.36 / 0.89 | 6.5 | 59 | `methods/impromptu_vla` |
| Qwen2.5-VL-7B, OpenEMMA-style | VLM, training-free | olp_vlm | – | yes | 5 | 1 | 0.33 / 0.77 | 6.2 | 23 | `methods/openemma_qwen` |

Notes:
- For VAD-Base, K counts the three command branches; the commanded one is `trajs[0]`.
- Hugging Face revisions behind these numbers: Qwen2.5-VL-7B-Instruct `cc59489`,
  Qwen2.5-VL-3B-Instruct `6628554`, timm resnet34.a1_in1k `0e87571`. `download_ckpts.sh` fetches
  the current `main` of each repo.

## 7. Notes for the formal evaluation

1. **Use one GT definition for every method, Alpamayo included.**
   - The ego_pose 180° fix is mandatory.
   - Choose between deduplicated poses (`pose_mode='dedup'`, used here) and the raw
     table. Measured on the 960 keyframes, the GT differs on average by 0.04 m over
     0–3 s; at 3 s the mean is 0.05 m, the p95 0.22 m and the max 0.33 m.
   - `labels/ego_trajectory.json` is 2 Hz and carries the flip in `*_traj_rel`.
2. **Horizons differ.** VAD, UniAD and Impromptu plan 3 s, the NAVSIM models 4 s, and the
   OpenEMMA-style planner and AutoVLA 5 s; Alpamayo plans 6.4 s. Compare at a common
   horizon of at most 3 s, or report each at its native horizon.
3. **K differs** (column K). min-over-K is not comparable across methods; use the top-1
   plan (`trajs[0]`) for the main table.
4. **Privileged and ego inputs** (see the cmd? / ego? columns).
   - The GT-derived command is used by VAD, UniAD, the NAVSIM models and AutoVLA. The
     OpenEMMA-style planner and Impromptu get no command.
   - Methods that receive past motion (NAVSIM ego status, Impromptu / OpenEMMA history)
     can largely extrapolate. That explains their low 1 s errors and their non-zero
     reverse recall.
5. **Reproducibility.** Checked by rerunning 80 keyframes (scenes 0057-0058) from a fresh
   install and comparing with our first run:
   - Identical: the kinematic baselines, the four NAVSIM models
     (DiffusionDrive samples with a fixed seed), VAD-Base, Impromptu-VLA and AutoVLA. UniAD
     differs by at most 0.1 mm.
   - Not bit-reproducible: Qwen2.5-VL-7B (OpenEMMA-style). Its long greedy fp16 generations
     can flip at near-tied tokens; two runs in the same env differed on 6 of the 80 frames.
     Over four runs, ADE/FDE@3s on these frames ranged 0.58-0.61 / 1.32-1.37 m, col%
     10.0-11.3 and revRec% 17-21 (4 or 5 of the 24 reversing frames). Report one run and
     state this, or average several runs.
6. **Full dataset.** Export `OLP_SCENES` / `OLP_TAG` / `OLP_OUTPUTS` (README, "Run") and rerun
   `run_recommended.sh`. It rebuilds the derived inputs for the new scene list
   (`tools/make_nusc_virtual.py`, `methods/navsim/make_navsim_virtual.py`) before inference.
