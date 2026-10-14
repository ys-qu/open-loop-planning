#!/usr/bin/env python
"""Build a "virtual nuScenes" view of the dataset keyframes for nuScenes-trained models
(VAD, UniAD; the front view is also used by the Qwen2.5-VL planners).

What it does
  1. Cameras: the dataset cameras are ~100 deg HFOV pinholes (f~1200 px @ 2880x1860).
     Each image is rescaled so its focal length equals that of the corresponding
     nuScenes camera, then cropped to 1600x900 around the nuScenes principal point.
     Extrinsics are unchanged, so the output is an exact pinhole camera with a
     nuScenes-like K (narrower FOV than the original image).
  2. LiDAR frame: nuScenes models work in the LIDAR_TOP frame (x right, y forward,
     z up, ~1.84 m above ground, 0.94 m ahead of the rear axle). We define a virtual
     LiDAR frame with that pose relative to the dataset ego frame; all *2lidar
     transforms in the infos refer to it. (The .pcd.bin path is the real sensor's and is
     in the real LiDAR frame - camera-only models never read it.)
  3. Ego pose: ego2global uses the corrected pose (ego_pose.json rotated 180 deg about z).
  4. can_bus (18): [pos(3), quat(4), accel(3, +9.81 on z), rot_rate(3), vel(3), 0, 0]
     from causal finite differences of the poses (the dataset has no CAN).
  5. command: nuScenes/VAD convention derived from the GT future (lateral offset at 3 s,
     +-2 m threshold) - the same privileged signal the nuScenes benchmark gives.

Output: <out>/samples/<NUSC_CAM>/*.jpg and <out>/infos_<tag>.json (list ordered by time).

  python tools/make_nusc_virtual.py --scenes scenes_subset.txt --tag subset
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OLP_DERIVED = os.environ.get('OLP_DERIVED', os.path.join(ROOT, 'derived'))
sys.path.insert(0, ROOT)
from olpcommon import DEFAULT_DATAROOT, GROUND_Z_EGO, NUSC_CAM_ORDER, ParkingDataset  # noqa: E402
from olpcommon.geometry import inv_T, make_T, rot_to_quat, rot_z  # noqa: E402

# nuScenes intrinsics (fx=fy, cx, cy) of the 6 cameras (scene-0061 calibration)
NUSC_K = {
    'CAM_FRONT': (1266.417, 816.267, 491.507),
    'CAM_FRONT_RIGHT': (1260.851, 807.968, 495.335),
    'CAM_FRONT_LEFT': (1272.598, 826.616, 479.751),
    'CAM_BACK': (809.221, 829.219, 481.773),
    'CAM_BACK_LEFT': (1256.748, 817.789, 451.954),
    'CAM_BACK_RIGHT': (1259.514, 807.253, 501.196),
}
OUT_W, OUT_H = 1600, 900
# virtual LIDAR_TOP pose in the dataset ego frame (nuScenes: x=0.94 m, z=1.84 m above ground)
VLIDAR2EGO = make_T(rot_z(-np.pi / 2), [0.94, 0.0, GROUND_Z_EGO + 1.84])
DEFAULT_OUT = os.path.join(_OLP_DERIVED, 'nusc_virtual')


def virtual_K(K, cam):
    f_t, cx_t, cy_t = NUSC_K[cam]
    s = f_t / K[0, 0]
    x0, y0 = s * K[0, 2] - cx_t, s * K[1, 2] - cy_t
    Kn = np.array([[s * K[0, 0], 0, s * K[0, 2] - x0], [0, s * K[1, 1], s * K[1, 2] - y0], [0, 0, 1.0]])
    return Kn, s, x0, y0


def render(job):
    src, dst, s, x0, y0 = job
    if os.path.exists(dst):
        return dst
    im = Image.open(src).convert('RGB')
    W, H = im.size
    im = im.resize((int(round(W * s)), int(round(H * s))), Image.BICUBIC if s > 1 else Image.LANCZOS)
    x0i, y0i = int(round(x0)), int(round(y0))
    canvas = Image.new('RGB', (OUT_W, OUT_H))
    canvas.paste(im.crop((x0i, y0i, x0i + OUT_W, y0i + OUT_H)), (0, 0))
    canvas.save(dst, quality=95)
    return dst


def T_to_rt(T):
    return T[:3, 3].tolist(), rot_to_quat(T[:3, :3]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--tag', default=os.environ.get('OLP_TAG', 'subset'))
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    db = ParkingDataset(args.dataroot, args.scenes, load_agents=False)
    for cam in NUSC_CAM_ORDER:
        os.makedirs(os.path.join(args.out, 'samples', cam), exist_ok=True)
    infos, jobs = [], []
    frame_idx, last_scene = 0, None
    for tok in db.keyframes:
        smp = db.sample[tok]
        frame_idx = 0 if smp['scene_token'] != last_scene else frame_idx + 1
        last_scene = smp['scene_token']
        e2g = db.ego2global(tok)
        lidar = db.lidar(tok)
        l2g = e2g @ VLIDAR2EGO
        g2l = inv_T(l2g)
        st = db.ego_state(tok)
        t, q = T_to_rt(e2g)
        can_bus = t + q + [st['ax'], st['ay'], 9.81, 0.0, 0.0, st['yaw_rate'], st['vx'], st['vy'], 0.0, 0.0, 0.0]
        cams = {}
        for cam in NUSC_CAM_ORDER:
            c = db.camera(tok, cam)
            Kn, s, x0, y0 = virtual_K(c['K'], cam)
            dst = os.path.join(args.out, 'samples', cam, os.path.basename(c['path']))
            jobs.append((c['path'], dst, s, x0, y0))
            c2l = g2l @ c['ego2global'] @ c['sensor2ego']
            ct, cq = T_to_rt(c['sensor2ego'])
            gt_, gq = T_to_rt(c['ego2global'])
            cams[cam] = {
                'data_path': dst, 'orig_path': c['path'], 'type': cam, 'src_channel': c['channel'],
                'sample_data_token': c['token'], 'timestamp': c['timestamp'],
                'cam_intrinsic': Kn.tolist(), 'width': OUT_W, 'height': OUT_H,
                'scale': s, 'crop_xy': [x0, y0],
                'sensor2ego_translation': ct, 'sensor2ego_rotation': cq,
                'ego2global_translation': gt_, 'ego2global_rotation': gq,
                'sensor2lidar_rotation': c2l[:3, :3].tolist(), 'sensor2lidar_translation': c2l[:3, 3].tolist(),
            }
        # ego past / future (virtual LiDAR origin, in the current virtual LiDAR frame)
        def lidar_xy(offsets):
            out, valid = [], []
            for off in offsets:
                T = db.ego2global_at(db.timestamp(tok) + int(round(off * 1e6)))
                if T is None:
                    out.append([0.0, 0.0])
                    valid.append(False)
                    continue
                p = (g2l @ T @ VLIDAR2EGO)[:3, 3]
                out.append(p[:2].tolist())
                valid.append(True)
            return out, valid
        his, his_valid = lidar_xy([-1.0, -0.5, 0.0])
        fut, fut_valid = lidar_xy([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        lt, lq = T_to_rt(VLIDAR2EGO)
        infos.append({
            'token': tok, 'scene_token': smp['scene_token'], 'scene_name': db.scene_name(tok),
            'frame_idx': frame_idx, 'timestamp': smp['timestamp'],
            'prev': db.prev_keyframe(tok) or '', 'next': db.next_keyframe(tok) or '',
            'lidar_path': lidar['path'],
            'lidar2ego_translation': lt, 'lidar2ego_rotation': lq,
            'ego2global_translation': t, 'ego2global_rotation': q,
            'cams': cams, 'can_bus': can_bus, 'ego_state': st,
            'command': db.command(tok), 'direction': db.direction(tok),
            'ego_his_lidar_xy': his, 'ego_his_valid': his_valid,
            'ego_fut_lidar_xy': fut, 'ego_fut_valid': fut_valid,
        })
    meta = {'vlidar2ego': VLIDAR2EGO.tolist(), 'nusc_K': NUSC_K, 'dataroot': args.dataroot,
            'scenes': db.scene_names}
    p = os.path.join(args.out, 'infos_%s.json' % args.tag)
    with open(p, 'w') as f:
        json.dump({'meta': meta, 'infos': infos}, f)
    print('wrote', p, len(infos), 'keyframes; rendering', len(jobs), 'images')
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(render, jobs, chunksize=8)):
            if i % 1000 == 0:
                print('  rendered', i)
    print('done')


if __name__ == '__main__':
    main()
