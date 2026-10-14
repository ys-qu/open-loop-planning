#!/usr/bin/env python
"""Frame / axis sanity check for prediction files (numpy only, base python is fine).

For frames where the ego clearly drives forward (GT x(4 s) > 2 m) it reports the share of top-1
plans that also go forward (x(4 s) > 0.5 m); for frames that clearly turn (|GT y(4 s)| > 2 m) the
share whose top-1 lateral offset has the same sign as the GT (a y-flip would give ~0%).

  python methods/navsim/axis_check.py --pred "outputs/navsim_*/predictions.json"
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from olpcommon import DEFAULT_DATAROOT, ParkingDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', nargs='+', required=True)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    args = ap.parse_args()
    db = ParkingDataset(DEFAULT_DATAROOT, args.scenes, load_agents=False, verbose=False)
    gt = {}
    for tok in db.keyframes:
        g = db.gt_future(tok, 4.0, 0.5)
        if g['valid'].all():
            gt[tok] = g['xy']
    fwd = [t for t, xy in gt.items() if xy[-1, 0] > 2.0]
    turn = [t for t, xy in gt.items() if xy[-1, 0] > 2.0 and abs(xy[-1, 1]) > 2.0]
    print('frames: forward %d, forward+turning %d' % (len(fwd), len(turn)))
    print('%-32s %10s %12s %14s' % ('variant', 'fwd->fwd%', 'turn sign%', 'median |dy4| m'))
    for f in sorted({f for p in args.pred for f in glob.glob(p)}):
        P = json.load(open(f))['predictions']

        def at4(tok):
            e = P[tok]
            t = np.asarray(e['t'])
            tr = np.asarray(e['trajs'][0])
            return np.array([np.interp(4.0, t, tr[:, 0]), np.interp(4.0, t, tr[:, 1])])
        ff = np.mean([at4(t)[0] > 0.5 for t in fwd if t in P]) * 100
        ts = np.mean([np.sign(at4(t)[1]) == np.sign(gt[t][-1, 1]) for t in turn if t in P]) * 100
        dy = np.median([abs(at4(t)[1] - gt[t][-1, 1]) for t in turn if t in P])
        print('%-32s %10.1f %12.1f %14.2f' % (os.path.basename(os.path.dirname(f)), ff, ts, dy))


if __name__ == '__main__':
    main()
