#!/usr/bin/env python
"""Zero-shot UniAD 2.0 (OpenDriveLab/UniAD, nuScenes base e2e checkpoint) on the parking dataset.

Inputs come from the virtual nuScenes view (tools/make_nusc_virtual.py): 6 cameras at
nuScenes intrinsics (1600x900, used at full resolution like UniAD), virtual LIDAR_TOP
frame, can_bus and lidar2global from the corrected poses, nuScenes-style command derived
from the GT future (as in the nuScenes protocol). model.forward_test is called directly,
frame by frame in time order (tracking / BEV state reset at scene boundaries, like
UniAD's own test loop). GT-only arguments that forward_test needs for its internal
metrics are filled with dummies (they do not influence predictions).

Output: outputs/uniad_base/predictions.json - 1 trajectory (after UniAD's occupancy-based
collision optimisation), 6 waypoints @ 2 Hz, converted to the ego origin. The raw
(pre-optimisation) plan is stored in extra.

  CUDA_VISIBLE_DEVICES=1 flock locks/gpu1.lock \
    $CONDA_BASE/envs/olp_uniad/bin/python methods/uniad/infer.py
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
UNIAD_DIR = os.path.join(ROOT, 'third_party', 'UniAD')
sys.path.insert(0, ROOT)
sys.path.insert(0, UNIAD_DIR)

import mmcv  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from mmdet3d.core.bbox import LiDARInstance3DBoxes  # noqa: E402

from olpcommon import NUSC_CAM_ORDER, save_predictions  # noqa: E402
from olpcommon.geometry import quat_to_rot, ref_traj_to_ego, yaw_of  # noqa: E402
from olpcommon.predictions import make_entry  # noqa: E402

CMD_IDX = {'RIGHT': 0, 'LEFT': 1, 'STRAIGHT': 2}  # UniAD: 0-Right, 1-Left, 2-Forward


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


def build_inputs(info, norm):
    imgs, l2i, shapes = [], [], []
    for cam in NUSC_CAM_ORDER:
        c = info['cams'][cam]
        img = mmcv.imread(c['data_path'], 'unchanged').astype(np.float32)
        img = mmcv.imnormalize(img, np.array(norm['mean'], np.float32), np.array(norm['std'], np.float32),
                               norm['to_rgb'])
        img = mmcv.impad_to_multiple(img, 32, pad_val=0)
        imgs.append(img)
        shapes.append(img.shape)
        l2i.append(lidar2img_from_cam(c))
    img = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).float()[None]
    q = info['ego2global_rotation']
    can_bus = np.array(info['can_bus'], dtype=np.float64)
    can_bus[:3] = info['ego2global_translation']
    can_bus[3:7] = q
    patch_angle = np.degrees(yaw_of(quat_to_rot(q)))
    if patch_angle < 0:
        patch_angle += 360
    can_bus[-2] = patch_angle / 180 * np.pi
    can_bus[-1] = patch_angle
    meta = dict(can_bus=can_bus, lidar2img=l2i, img_shape=shapes, pad_shape=shapes, ori_shape=shapes,
                scale_factor=1.0, scene_token=info['scene_token'], box_type_3d=LiDARInstance3DBoxes,
                sample_idx=info['token'], pts_filename=info['lidar_path'])
    # lidar -> global (UniAD row-vector convention: p_g = p_l @ R + t)
    l2e_r = quat_to_rot(info['lidar2ego_rotation'])
    l2e_t = np.array(info['lidar2ego_translation'])
    e2g_r = quat_to_rot(info['ego2global_rotation'])
    e2g_t = np.array(info['ego2global_translation'])
    l2g_r_mat = l2e_r.T @ e2g_r.T
    l2g_t = l2e_t @ e2g_r.T + e2g_t
    # UniAD 2.0 also reads l2g_r_mat (same row-vector matrix) from img_metas for the BEV shift
    meta['l2g_r_mat'] = l2g_r_mat.astype(np.float32)
    return img, meta, l2g_r_mat, l2g_t


def dummy_gt(device):
    T = 10
    return dict(
        gt_lane_labels=[torch.tensor([[0, 3]], device=device)],
        gt_lane_masks=[torch.zeros(1, 2, 200, 200, dtype=torch.long, device=device)],
        gt_segmentation=[torch.zeros(1, T, 200, 200, device=device)],
        gt_instance=[torch.zeros(1, T, 200, 200, device=device)],
        gt_occ_img_is_valid=[torch.ones(1, T, device=device)],
        sdc_planning=[torch.zeros(1, 1, 6, 3, device=device)],
        sdc_planning_mask=[torch.zeros(1, 1, 6, 2, device=device)],
    )


def lidar_xy_to_ego(xy_l, vlidar2ego):
    R = np.array(vlidar2ego)[:2, :2]
    return np.asarray(xy_l) @ R.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='projects/configs/stage2_e2e/base_e2e.py')
    ap.add_argument('--ckpt', default=os.path.join(ROOT, 'ckpts/uniad/uniad_base_e2e.pth'))
    ap.add_argument('--infos', default=os.path.join(_OLP_DERIVED, 'nusc_virtual', 'infos_%s.json' % os.environ.get('OLP_TAG', 'subset')))
    ap.add_argument('--out', default=os.path.join(os.environ.get('OLP_OUTPUTS', os.path.join(ROOT, 'outputs')), 'uniad_base', 'predictions.json'))
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--det-thresh', type=float, default=0.3)
    args = ap.parse_args()

    args.out, args.infos, args.ckpt = map(os.path.abspath, (args.out, args.infos, args.ckpt))
    os.chdir(UNIAD_DIR)
    import projects.mmdet3d_plugin  # noqa: F401  (registers UniAD modules)
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model = model.cuda().eval()
    norm = cfg.img_norm_cfg

    d = json.load(open(args.infos))
    infos, vl2e = d['infos'], d['meta']['vlidar2ego']
    ref_offset = np.array(vl2e)[:2, 3]
    if args.max_frames:
        infos = infos[:args.max_frames]
    preds = {}
    t0 = time.time()
    for i, info in enumerate(infos):
        img, meta, l2g_r, l2g_t = build_inputs(info, norm)
        cmd = info['command']
        with torch.no_grad():
            res = model.forward_test(
                img=[img.cuda()], img_metas=[[meta]],
                l2g_t=torch.tensor(l2g_t, dtype=torch.float32, device='cuda'),
                l2g_r_mat=torch.tensor(l2g_r, dtype=torch.float32, device='cuda'),
                timestamp=[torch.tensor(info['timestamp'] / 1e6, dtype=torch.float64, device='cuda')],
                command=torch.tensor([CMD_IDX[cmd]], device='cuda'),
                rescale=True, **dummy_gt('cuda'))
        rp = res[0]['planning']['result_planning']
        traj_l = rp['sdc_traj'][0].float().cpu().numpy()[:, :2]  # 6x2, positions rel. to LiDAR origin
        traj = ref_traj_to_ego(lidar_xy_to_ego(traj_l, vl2e), ref_offset)
        raw_all = rp.get('sdc_traj_all')
        raw = None
        if raw_all is not None:
            raw_l = raw_all[0].float().cpu().numpy()[:, :2]
            raw = np.round(ref_traj_to_ego(lidar_xy_to_ego(raw_l, vl2e), ref_offset), 3).tolist()
        # detections (lidar frame -> ego frame) for qualitative checks
        dets = []
        if 'boxes_3d' in res[0]:
            b = res[0]['boxes_3d']
            s = res[0]['scores_3d'].cpu().numpy()
            l = res[0]['labels_3d'].cpu().numpy()
            bt = b.tensor.cpu().numpy() if hasattr(b, 'tensor') else np.asarray(b)
            R = np.array(vl2e)[:3, :3]
            for bb, ss, ll in zip(bt[s > args.det_thresh], s[s > args.det_thresh], l[s > args.det_thresh]):
                c = R @ bb[:3] + np.array(vl2e)[:3, 3]
                # mmdet3d 1.0 LiDAR boxes: dims (l, w, h), yaw = heading
                yaw = float(bb[6]) + float(np.arctan2(R[1, 0], R[0, 0]))
                dets.append([round(float(c[0]), 2), round(float(c[1]), 2), round(float(bb[3]), 2),
                             round(float(bb[4]), 2), round(float(np.arctan2(np.sin(yaw), np.cos(yaw))), 3),
                             round(float(ss), 3), int(ll)])
        preds[info['token']] = make_entry(np.arange(1, 7) * 0.5, traj[None], scores=[1.0],
                                          extra={'command': cmd, 'raw_plan_before_col_optim': raw,
                                                 'ref_traj_vlidar_xy': np.round(traj_l, 4).tolist(),
                                                 'dets_ego[x,y,l,w,yaw,score,label]': dets})
        if i % 50 == 0:
            print('[%d/%d] %.2fs/frame cmd=%s end=%s ndet=%d' % (i, len(infos), (time.time() - t0) / (i + 1),
                                                                  cmd, np.round(traj[-1], 2), len(dets)), flush=True)
    save_predictions(args.out, 'uniad_base', preds, info={
        'model': 'UniAD 2.0 base e2e (R101)', 'checkpoint': args.ckpt, 'config': args.config,
        'inputs': '6 virtual nuScenes cams 1600x900, can_bus + lidar2global from poses, GT-derived command',
        'temporal': True, 'K': 1, 'horizon_s': 3.0, 'collision_optim': True,
        'ref_point': 'virtual LiDAR (0.94 m ahead of ego origin) converted to ego origin'})


if __name__ == '__main__':
    main()
