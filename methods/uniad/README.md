# UniAD 2.0 (base e2e, R101) — zero-shot on the parking dataset

> **Setup (shared with all methods)**, from the repo root after `source env.sh`:
> - code: `bash setup/fetch_third_party.sh`
> - env: `bash setup/create_env.sh olp_uniad`; `methods/uniad/setup_env.sh` does the same
> - weights: `bash setup/download_ckpts.sh uniad`
> - derived inputs: top-level README, "Run a single method"
>
> Paths below are relative to the repo root. `$OLP_DERIVED` and `$CONDA_BASE` are
> set by `env.sh`.

- **Repo:** github.com/OpenDriveLab/UniAD, branch `v2.0` @ `609ee08`, cloned to `third_party/UniAD`.
- **Checkpoint:** HF `OpenDriveLab/UniAD2.0_R101_nuScenes`, file `ckpts/uniad_base_e2e.pth`,
  stored in `ckpts/uniad/`.
- **Motion anchors:** same HF repo, file `data/motion_anchor_infos_mode6.pkl`, stored in
  `ckpts/uniad/` and symlinked to `third_party/UniAD/data/others/` (the config expects it
  there).

```bash
mkdir -p ckpts/uniad && cd ckpts/uniad
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_e2e.pth
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/data/motion_anchor_infos_mode6.pkl
mkdir -p ../../third_party/UniAD/data/others
ln -sf ../../../../ckpts/uniad/motion_anchor_infos_mode6.pkl ../../third_party/UniAD/data/others/
```

## Environment

`bash methods/uniad/setup_env.sh` creates conda env `olp_uniad`: python 3.9,
torch 2.0.1+cu118, mmcv-full 1.7.2, mmdet 2.26.0, mmseg 0.29.1, mmdet3d 1.0.0rc6.

Deviations from `third_party/UniAD/docs/INSTALL.md`:
- mmcv-full 1.7.2 instead of 1.6.1, because it is the only prebuilt cu118/torch2.0
  wheel. The script relaxes mmdet3d's upper version check accordingly.
- matplotlib 3.7.5 and numba 0.56.4 are pinned, because newer wheels need a newer
  numpy than UniAD's pin.

The checkpoint loads with no missing or unexpected keys.

## Inputs

The virtual nuScenes view from `tools/make_nusc_virtual.py` (see `docs/notes.md` §2):
- 6 cameras at nuScenes intrinsics, 1600×900, used at full resolution, normalised the
  caffe way and padded to 1600×928, like UniAD's test pipeline.
- `lidar2img`, `can_bus`, and `l2g_r_mat` / `l2g_t` in UniAD's row-vector convention,
  built from the virtual LIDAR_TOP frame and the corrected ego poses. UniAD 2.0 reads
  `l2g_r_mat` both as an argument and from `img_metas`; both are filled.
- `timestamp` as float64 seconds, needed for track propagation.
- `command`: 0 = right, 1 = left, 2 = straight, **derived from the GT future** as in the
  nuScenes protocol. This is the only privileged input.

`model.forward_test` is called directly, one keyframe at a time in time order, so
UniAD's own temporal state is kept: BEV and track queries carry over, and reset when the
scene changes.

`forward_test` also takes GT inputs that only feed its internal nuScenes metrics (lane
masks, occupancy GT, `sdc_planning`). These get zero dummies; they do not affect the
predictions.

## Run

```bash
cd $OLP_REPO
CUDA_VISIBLE_DEVICES=0 flock locks/gpu0.lock \
  $CONDA_BASE/envs/olp_uniad/bin/python methods/uniad/infer.py
# optional: --max-frames N, --out <path>, --infos <virtual infos json>
```

- Runtime: about 1.4 s per frame, roughly 23 min for 960 frames.
- GPU memory: about 7 GB.

## Output

`outputs/uniad_base/predictions.json`:
- K = 1: the final plan, after UniAD's occupancy-based collision optimisation.
- 6 waypoints at 0.5 … 3.0 s.
- UniAD plans the LiDAR origin's future positions in the LiDAR frame (x right,
  y forward). The script converts them to the ego frame and the ego origin.
- `extra.raw_plan_before_col_optim`: the plan before the collision optimisation.
- `extra.dets_ego`: UniAD's 3D detections in the ego frame, as
  `[x, y, length, width, yaw, score, label]`.

## Sanity metrics

See `docs/notes.md` §6.

## Checks done

- **Detections line up with the human labels.** Parked rows and their orientation match
  in `tools/visualize.py`. This validates the whole virtual-camera / lidar2img chain.
- **Typical failure** (render it with `tools/visualize.py --pred outputs/uniad_base/predictions.json --tokens a2df6997`): the ego sits in a
  stall and the GT is almost stationary, yet UniAD plans 22 m straight ahead through the
  aisle into the opposite row.
