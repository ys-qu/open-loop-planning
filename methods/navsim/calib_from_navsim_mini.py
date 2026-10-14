#!/usr/bin/env python
"""Derive the nominal nuPlan / NAVSIM sensor model from the local NAVSIM mini split.

Writes methods/navsim/olp_navsim/nuplan_cam_model.json with
  * camera intrinsics K, distortion D (OpenCV 5-param), image size (most common values)
  * mean sensor2ego rotation / translation of every NAVSIM camera (lidar2ego is the
    identity in OpenScene, so sensor2lidar == sensor2ego)
  * ground height of the merged LiDAR point cloud in the nuPlan ego (rear-axle) frame

Only needs numpy (+ the OpenScene pickles / pcds), e.g.
  python -I methods/navsim/calib_from_navsim_mini.py --navsim-root $NAVSIM_MINI_ROOT
"""
import argparse
import collections
import glob
import json
import os
import pickle

import numpy as np

CAMS = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2']
HERE = os.path.dirname(os.path.abspath(__file__))


def read_pcd_xyz(path):
    with open(path, 'rb') as f:
        data = f.read()
    i = data.index(b'DATA binary\n') + len(b'DATA binary\n')
    hdr = data[:i].decode()
    n = int([l for l in hdr.split('\n') if l.startswith('POINTS')][0].split()[1])
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', 'u1'), ('lidar_info', 'u1'), ('ring', 'u1')])
    a = np.frombuffer(data[i:i + n * 15], dtype=dt)
    return np.stack([a['x'], a['y'], a['z']], 1)


def mean_rotation(Rs):
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--navsim-root', default=os.environ.get('NAVSIM_MINI_ROOT', ''))
    ap.add_argument('--split', default='mini')
    ap.add_argument('--out', default=os.path.join(HERE, 'olp_navsim', 'nuplan_cam_model.json'))
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(args.navsim_root, 'navsim_logs', args.split, '*.pkl')))
    Rs, ts = collections.defaultdict(list), collections.defaultdict(list)
    KD = collections.Counter()
    pcds = []
    for lp in logs:
        d = pickle.load(open(lp, 'rb'))
        for f in d[::20]:
            assert np.allclose(np.asarray(f['lidar2ego']), np.eye(4)), 'lidar2ego is not identity'
            for c in CAMS:
                cam = f['cams'][c]
                Rs[c].append(np.asarray(cam['sensor2lidar_rotation'], dtype=np.float64))
                ts[c].append(np.asarray(cam['sensor2lidar_translation'], dtype=np.float64))
                K = np.asarray(cam['cam_intrinsic'], dtype=np.float64)
                D = np.asarray(cam['distortion'], dtype=np.float64)
                KD[tuple(np.round(K.ravel(), 3)) + tuple(np.round(D, 5))] += 1
        pcds.append(os.path.join(args.navsim_root, 'sensor_blobs', args.split, d[len(d) // 2]['lidar_path']))

    kd = np.array(KD.most_common(1)[0][0])
    K, D = kd[:9].reshape(3, 3), kd[9:]
    cams = {}
    for c in CAMS:
        R = mean_rotation(np.array(Rs[c]))
        ax = R[:, 2]
        cams[c] = {'sensor2ego_rotation': R.tolist(), 'sensor2ego_translation': np.mean(ts[c], 0).tolist(),
                   'yaw_deg': float(np.degrees(np.arctan2(ax[1], ax[0]))),
                   'pitch_deg': float(np.degrees(np.arcsin(ax[2]))), 'n': len(Rs[c])}

    modes = []
    for p in pcds:
        xyz = read_pcd_xyz(p)
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        z = xyz[(r > 4) & (r < 20) & (xyz[:, 2] < 0.5) & (xyz[:, 2] > -1.5), 2]
        h, e = np.histogram(z, bins=np.arange(-1.5, 0.5, 0.02))
        modes.append(e[h.argmax()] + 0.01)
    out = {'source': 'NAVSIM %s split (%d logs), %s' % (args.split, len(logs), args.navsim_root),
           'width': 1920, 'height': 1080, 'K': K.tolist(), 'D': D.tolist(),
           'KD_counts': {str(k): v for k, v in KD.most_common()},
           'ground_z_ego': float(np.median(modes)), 'ground_z_ego_p10_p90': np.percentile(modes, [10, 90]).tolist(),
           'cams': cams}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)
    print('K', K.tolist(), 'D', D.tolist(), 'ground z %.3f' % out['ground_z_ego'])
    for c, v in cams.items():
        print('%s yaw %.2f pitch %.2f t %s' % (c, v['yaw_deg'], v['pitch_deg'], np.round(v['sensor2ego_translation'], 3)))


if __name__ == '__main__':
    main()
