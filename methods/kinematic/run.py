#!/usr/bin/env python
"""Non-learned reference planners: stationary, constant velocity (CV), and constant
speed + yaw rate (CTRV). Inputs: past ego poses only (causal). Output: 6.4 s at 10 Hz.

  python methods/kinematic/run.py --scenes scenes_subset.txt --out-dir outputs
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from olpcommon import DEFAULT_DATAROOT, ParkingDataset, save_predictions  # noqa: E402
from olpcommon.predictions import make_entry  # noqa: E402


def ctrv(speed, yaw_rate, t):
    if abs(yaw_rate) < 1e-3:
        return np.stack([speed * t, np.zeros_like(t)], 1)
    r = speed / yaw_rate
    return np.stack([r * np.sin(yaw_rate * t), r * (1 - np.cos(yaw_rate * t))], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    ap.add_argument('--out-dir', default=os.environ.get('OLP_OUTPUTS', os.path.join(ROOT, 'outputs')))
    ap.add_argument('--horizon', type=float, default=6.4)
    args = ap.parse_args()

    db = ParkingDataset(args.dataroot, args.scenes, load_agents=False)
    t = np.round(np.arange(1, int(round(args.horizon / 0.1)) + 1) * 0.1, 3)
    preds = {'stationary': {}, 'const_vel': {}, 'ctrv': {}}
    for tok in db.keyframes:
        s = db.ego_state(tok)
        v = np.array([s['vx'], s['vy']])
        preds['stationary'][tok] = make_entry(t, np.zeros((len(t), 2)))
        preds['const_vel'][tok] = make_entry(t, v[None] * t[:, None])
        spd = s['speed']
        w = float(np.clip(s['yaw_rate'], -0.6, 0.6))
        preds['ctrv'][tok] = make_entry(t, ctrv(spd, w, t))
    for name, p in preds.items():
        save_predictions(os.path.join(args.out_dir, name, 'predictions.json'), name, p,
                         info={'inputs': 'causal ego state from ego_pose (dt=0.2 s)', 'horizon_s': args.horizon})


if __name__ == '__main__':
    main()
