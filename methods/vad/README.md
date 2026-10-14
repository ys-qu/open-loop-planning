# VAD-Base: zero-shot on the parking dataset

> **Setup (shared with all methods)**, from the repo root after `source env.sh`:
> - code: `bash setup/fetch_third_party.sh`
> - env: `bash setup/create_env.sh olp_uniad`; `methods/vad/setup_env.sh` does the same
> - weights: `bash setup/download_ckpts.sh vad`
> - derived inputs: top-level README, "Run a single method"
>
> Paths below are relative to the repo root. `$OLP_DERIVED` and `$CONDA_BASE` are set by
> `env.sh`.

| | |
|---|---|
| Method | VAD (vectorized scene representation for end-to-end driving), Base model, trained on nuScenes |
| Code | github.com/hustvl/VAD @ `1688c4b`, cloned to `third_party/VAD` |
| Checkpoint | Google Drive id `1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS` (VAD README "VAD-Base"), saved as `ckpts/vad/VAD_base.pth` |
| Env | `olp_uniad`, shared with UniAD: python 3.9, torch 2.0.1+cu118, mmcv-full 1.7.2, mmdet 2.26.0, mmdet3d 1.0.0rc6 |
| Runner | `methods/vad/infer.py` |
| Output | `outputs/vad_base/predictions.json` |

## Run

```bash
source env.sh
CUDA_VISIBLE_DEVICES=0 $CONDA_BASE/envs/olp_uniad/bin/python methods/vad/infer.py
```

Options:
- `--infos <file>`: default `$OLP_DERIVED/nusc_virtual/infos_$OLP_TAG.json`.
- `--out <file>`.
- `--max-frames N`: smoke test.

Runtime is about 0.9 s per frame on a Quadro RTX 6000; most of it is CPU image loading and
resizing. Peak GPU memory is about 5 GB.

## Inputs

Everything below is built by `tools/make_nusc_virtual.py` (see `docs/notes.md` §2).

- **Six virtual nuScenes cameras.** 1600×900 at nuScenes intrinsics; the model then
  normalises, scales by 0.8 and pads to /32, as in VAD's test pipeline.
- **Virtual nuScenes LIDAR_TOP frame.** All `lidar2img` matrices and the outputs refer to
  it.
- **can_bus.** Built from the corrected poses, because the dataset has no CAN data.
- **`ego_fut_cmd`.** The nuScenes-style one-hot [right, left, straight], **derived from
  the GT future** (±2 m lateral at 3 s). This is the privileged input of the nuScenes
  protocol.
- **Unused ego status.** `ego_his_trajs` / `ego_lcf_feat` are filled from causal pose
  history, but the VAD-Base config does not use them.

VAD's own `forward_test` is called for each keyframe in time order, so its BEV features are
carried from frame to frame and reset when the scene changes. The nuScenes-metric helpers
inside `forward_test` are replaced by no-ops, and dummy GT tensors are passed where its
signature requires them.

**Deviation from the official setup.** VAD targets torch 1.9 / mmcv-full 1.4 /
mmdet3d 0.17.1. Here it runs in the UniAD 2.0 env (mmdet3d 1.0) with a single
compatibility shim: `mmdet3d.ops.roiaware_pool3d` → `mmcv.ops.points_in_boxes_*`, which
inference never calls. The runner checks that the checkpoint initialises every
parameter: missing 0, unexpected 0.

## Output

- **Trajectories.** `t` = 0.5 … 3.0 s, three command branches with the GT-derived command
  first (`trajs[0]`); `extra.branches` names them.
- **Coordinates.** VAD predicts per-step offsets of the LiDAR origin in the LiDAR frame (x
  right, y forward). They are summed, rotated to the ego frame and moved from the virtual
  LiDAR (0.94 m ahead) to the ego origin. The raw plan is kept in
  `extra.ref_traj_vlidar_xy`.
- **Perception.** `extra.dets_ego` holds VAD's detections as
  `[x, y, length, width, yaw, score, label]`, and `extra.map_ego` its map polylines, both
  in the ego frame. `tools/visualize.py` draws them.

## Notes

- VAD's perception transfers poorly here: detections often miss parked cars, and the map
  head draws road boundaries inside the parking lot.
- When the ego is stopped, VAD mostly predicts staying still. It never predicts reversing.
