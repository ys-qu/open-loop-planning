#!/usr/bin/env python
"""Zero-shot VAD-Base (hustvl/VAD, nuScenes checkpoint) on the dataset keyframes.

VAD's own `forward_test` is called frame by frame in time order, so its test-time logic is
kept (BEV features are propagated between consecutive keyframes and reset when the scene
changes). The metric helpers that forward_test calls with nuScenes GT
(`assign_pred_to_gt_vip3d`, `compute_motion_metric_vip3d`, `compute_planner_metric_stp3`)
are replaced by no-ops, and dummy GT tensors are passed where the signature requires them.

Inputs: virtual nuScenes view from tools/make_nusc_virtual.py (6 cams at nuScenes
intrinsics, virtual LIDAR_TOP frame, can_bus from poses, nuScenes-style command derived
from the GT future - the privileged input of the nuScenes protocol). VAD-Base does not use
the ego-status inputs (ego_his_trajs / ego_lcf_feat); they are filled from causal pose
history anyway.

  CUDA_VISIBLE_DEVICES=0 $CONDA_BASE/envs/olp_uniad/bin/python methods/vad/infer.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OLP_DERIVED = os.environ.get('OLP_DERIVED', os.path.join(ROOT, 'derived'))
sys.path.insert(0, ROOT)

REPO = 'VAD'
CONFIG = 'projects/configs/VAD/VAD_base_stage_2.py'
CKPT = 'ckpts/vad/VAD_base.pth'
SCALE = 0.8                                        # VAD-Base test pipeline: 1600x900 -> 1280x720
CMD_IDX = {'RIGHT': 0, 'LEFT': 1, 'STRAIGHT': 2}  # VAD one-hot order
CMD_NAMES = ['RIGHT', 'LEFT', 'STRAIGHT']


def install_mmdet3d_compat():
    """VAD targets mmdet3d 0.17.1; in mmdet3d 1.0 the roiaware_pool3d ops live in mmcv.ops
    (only used by box utilities that inference never calls)."""
    import types
    try:
        import mmdet3d.ops.roiaware_pool3d  # noqa: F401
    except ImportError:
        import mmcv.ops
        shim = types.ModuleType('mmdet3d.ops.roiaware_pool3d')
        shim.points_in_boxes_gpu = mmcv.ops.points_in_boxes_part
        shim.points_in_boxes_cpu = mmcv.ops.points_in_boxes_cpu
        shim.points_in_boxes_batch = mmcv.ops.points_in_boxes_all
        sys.modules['mmdet3d.ops.roiaware_pool3d'] = shim


def perception_to_ego(r, vl2e, det_thresh, map_thresh):
    """Detections / map polylines (virtual LiDAR frame, mmdet3d-0.17 box convention:
    dims (w, l, h), yaw = -heading - pi/2) -> ego frame, for qualitative checks."""
    out = {}
    R, t = vl2e[:3, :3], vl2e[:3, 3]
    rot = np.arctan2(R[1, 0], R[0, 0])
    if 'boxes_3d' in r and 'scores_3d' in r:
        b = r['boxes_3d']
        bt = b.tensor.cpu().numpy() if hasattr(b, 'tensor') else np.asarray(b)
        sc = r['scores_3d'].cpu().numpy()
        lb = r['labels_3d'].cpu().numpy()
        dets = []
        for bb, s_, l_ in zip(bt[sc > det_thresh], sc[sc > det_thresh], lb[sc > det_thresh]):
            c = R @ bb[:3] + t
            yaw = -float(bb[6]) - np.pi / 2 + rot
            dets.append([round(float(c[0]), 2), round(float(c[1]), 2), round(float(bb[4]), 2), round(float(bb[3]), 2),
                         round(float(np.arctan2(np.sin(yaw), np.cos(yaw))), 3), round(float(s_), 3), int(l_)])
        out['dets_ego[x,y,l,w,yaw,score,label]'] = dets
    if 'map_pts_3d' in r and 'map_scores_3d' in r:
        classes = ['divider', 'ped_crossing', 'boundary']
        maps = []
        ms = r['map_scores_3d'].cpu().numpy()
        for pts, s_, l_ in zip(r['map_pts_3d'].cpu().numpy()[ms > map_thresh], ms[ms > map_thresh],
                               r['map_labels_3d'].cpu().numpy()[ms > map_thresh]):
            e = np.asarray(pts)[:, :2] @ R[:2, :2].T + t[:2]
            maps.append({'cls': classes[int(l_)] if int(l_) < 3 else str(int(l_)), 'score': round(float(s_), 3),
                         'pts': np.round(e, 2).tolist()})
        out['map_ego'] = maps
    return out


def lidar2img_from_cam(cam):
    K = np.eye(4)
    K[:3, :3] = np.array(cam['cam_intrinsic'])
    R = np.array(cam['sensor2lidar_rotation'])
    t = np.array(cam['sensor2lidar_translation'])
    lidar2cam_r = np.linalg.inv(R)
    lidar2cam_t = t @ lidar2cam_r.T
    rt = np.eye(4)
    rt[:3, :3] = lidar2cam_r.T
    rt[3, :3] = -lidar2cam_t
    return K @ rt.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='vad_base', choices=['vad_base'], help='output name')
    ap.add_argument('--infos', default=os.path.join(_OLP_DERIVED, 'nusc_virtual',
                                                    'infos_%s.json' % os.environ.get('OLP_TAG', 'subset')))
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--det-thresh', type=float, default=0.3)
    ap.add_argument('--map-thresh', type=float, default=0.4)
    args = ap.parse_args()
    out = os.path.abspath(args.out or os.path.join(os.environ.get('OLP_OUTPUTS', os.path.join(ROOT, 'outputs')),
                                                   args.model, 'predictions.json'))
    infos_path = os.path.abspath(args.infos)
    repo_dir = os.path.join(ROOT, 'third_party', REPO)
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)

    import mmcv
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from olpcommon import NUSC_CAM_ORDER, save_predictions
    from olpcommon.geometry import quat_to_rot, ref_traj_to_ego, yaw_of
    from olpcommon.predictions import make_entry
    install_mmdet3d_compat()
    import projects.mmdet3d_plugin  # noqa: F401  (registers the VAD modules)

    cfg = Config.fromfile(CONFIG)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    sd = torch.load(os.path.join(ROOT, CKPT), map_location='cpu')
    sd = sd.get('state_dict', sd)
    inc = model.load_state_dict(sd, strict=False)
    prefix = lambda keys: sorted({'.'.join(k.split('.')[:2]) for k in keys})  # noqa: E731
    print('[ckpt] missing %d %s | unexpected %d %s' % (len(inc.missing_keys), prefix(inc.missing_keys),
                                                       len(inc.unexpected_keys), prefix(inc.unexpected_keys)))
    if inc.missing_keys:
        raise RuntimeError('checkpoint does not initialise the whole model: %s' % prefix(inc.missing_keys))
    model = model.cuda().eval()
    for fn in ('assign_pred_to_gt_vip3d', 'compute_motion_metric_vip3d', 'compute_planner_metric_stp3'):
        setattr(model, fn, (lambda *a, **k: {}))
    norm = cfg.img_norm_cfg

    d = json.load(open(infos_path))
    infos, vl2e = d['infos'], np.array(d['meta']['vlidar2ego'])
    ref_offset = vl2e[:2, 3]
    if args.max_frames:
        infos = infos[:args.max_frames]
    S = np.eye(4)
    S[0, 0] = S[1, 1] = SCALE
    empty = LiDARInstance3DBoxes(torch.zeros(0, 9), box_dim=9)
    preds, t0 = {}, time.time()
    for i, info in enumerate(infos):
        imgs, l2i, shapes = [], [], []
        for cam in NUSC_CAM_ORDER:
            c = info['cams'][cam]
            img = mmcv.imread(c['data_path'], 'unchanged').astype(np.float32)
            img = mmcv.imnormalize(img, np.array(norm['mean'], np.float32), np.array(norm['std'], np.float32),
                                   norm['to_rgb'])
            h, w = img.shape[:2]
            img = mmcv.imresize(img, (int(w * SCALE), int(h * SCALE)), return_scale=False)
            img = mmcv.impad_to_multiple(img, 32, pad_val=0)
            imgs.append(img)
            shapes.append(img.shape)
            l2i.append(S @ lidar2img_from_cam(c))
        img = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).float()[None].cuda()
        q = info['ego2global_rotation']
        can_bus = np.array(info['can_bus'], dtype=np.float64)
        can_bus[:3] = info['ego2global_translation']
        can_bus[3:7] = q
        pa = np.degrees(yaw_of(quat_to_rot(q))) % 360
        can_bus[-2], can_bus[-1] = pa / 180 * np.pi, pa
        meta = dict(can_bus=can_bus, lidar2img=l2i, img_shape=shapes, pad_shape=shapes, ori_shape=shapes,
                    scene_token=info['scene_token'], box_type_3d=LiDARInstance3DBoxes, sample_idx=info['token'],
                    pts_filename=info['lidar_path'], scale_factor=1.0)
        cmd = info['command']
        onehot = torch.zeros(1, 1, 1, 3, device='cuda')
        onehot[..., CMD_IDX[cmd]] = 1
        # VAD-format ego history: per-step offsets of the LiDAR origin (2 steps), lidar frame
        his = np.array(info['ego_his_lidar_xy'])
        his_off = torch.tensor(np.diff(his, axis=0), dtype=torch.float32, device='cuda').view(1, 1, 2, 2)
        st = info['ego_state']
        lcf = torch.zeros(1, 1, 1, 9, device='cuda')
        # (vx, vy) in the lidar frame (x right, y fwd), ax, ay, yaw rate, ego length/width, speed, curvature
        lcf[0, 0, 0] = torch.tensor([-st['vy'], st['vx'], -st['ay'], st['ax'], st['yaw_rate'], 4.94, 2.0,
                                     st['speed'], 0.0])
        with torch.no_grad():
            res = model.forward_test(
                img_metas=[[meta]], gt_bboxes_3d=[[empty]], gt_labels_3d=[[torch.zeros(0, dtype=torch.long)]],
                img=[img], ego_his_trajs=[his_off], ego_fut_trajs=[torch.zeros(1, 1, 6, 2, device='cuda')],
                ego_fut_cmd=[onehot], ego_lcf_feat=[lcf], gt_attr_labels=[[torch.zeros(0, 34)]],
                fut_valid_flag=[[False]], ego_fut_masks=[torch.zeros(1, 1, 6, device='cuda')])
        r = res[0]['pts_bbox'] if 'pts_bbox' in res[0] else res[0]
        ef = r['ego_fut_preds'].float().cpu().numpy()   # [3 commands, 6 steps, 2] per-step offsets
        order = [CMD_IDX[cmd]] + [k for k in range(3) if k != CMD_IDX[cmd]]
        trajs, raw_l = [], []
        for k in order:
            xy_l = np.cumsum(ef[k][:, :2], axis=0)
            raw_l.append(np.round(xy_l, 4).tolist())
            trajs.append(ref_traj_to_ego(xy_l @ vl2e[:2, :2].T, ref_offset))
        # raw model output (virtual LiDAR origin, LiDAR frame x right / y fwd) for offline re-conversion
        extra = {'command': cmd, 'branches': [CMD_NAMES[k] for k in order], 'ref_traj_vlidar_xy': raw_l}
        extra.update(perception_to_ego(r, vl2e, args.det_thresh, args.map_thresh))
        preds[info['token']] = make_entry(np.arange(1, 7) * 0.5, np.stack(trajs), scores=[1.0, 0.0, 0.0], extra=extra)
        if i % 50 == 0:
            print('[%d/%d] %.2fs/frame cmd=%s top1 end=%s' % (i, len(infos), (time.time() - t0) / (i + 1), cmd,
                                                            np.round(trajs[0][-1], 2)), flush=True)
    save_predictions(out, args.model, preds, info={
        'model': 'VAD-Base', 'repo': REPO, 'config': CONFIG, 'checkpoint': CKPT, 'image_scale': SCALE,
        'inputs': '6 virtual nuScenes cams, can_bus from poses, GT-derived command',
        'temporal': True, 'K': '3 command branches (commanded first)', 'horizon_s': 3.0,
        'ckpt_check': {'missing': len(inc.missing_keys), 'unexpected': len(inc.unexpected_keys)},
        'ref_point': 'virtual LiDAR (0.94 m ahead of ego origin) converted to ego origin'})


if __name__ == '__main__':
    main()
