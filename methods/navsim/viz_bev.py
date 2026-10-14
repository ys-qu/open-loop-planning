#!/usr/bin/env python
"""Visual sanity check of NAVSIM-agent predictions on the dataset keyframes.

Per keyframe: the stitched TransFuser camera input (virtual cam_l0 | cam_f0 | cam_r0, exactly
the 1024x256 tensor the model sees) and a BEV with the LiDAR points the model sees (z > 0.2 m
in the nuPlan ego frame), the GT future (black), each variant's top-1 plan (and other modes,
thin), and optionally TransFuser's own agent detections (from run_agent.py --dump-aux).
BEV axes: x forward = up, y left = left.

  python methods/navsim/viz_bev.py --pred outputs/navsim_transfuser/predictions.json outputs/navsim_diffusiondrive/predictions.json \
      --aux logs/navsim_transfuser_aux.npz --out outputs/vis/navsim
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from olp_navsim import ROOT, OLPNavsimInputs  # noqa: E402
from olpcommon import DEFAULT_DATAROOT, EGO_CENTER_X, ParkingDataset, ego_box_corners  # noqa: E402
from olpcommon.geometry import box_corners_2d  # noqa: E402

COLORS = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan', 'tab:brown', 'tab:pink',
          'tab:olive', 'gold', 'navy', 'lime']


def pick_tokens(db, n_fwd=4):
    """A few informative frames: moving forward (spread over scenes), turning, reversing."""
    fwd, turn, rev = [], [], []
    for tok in db.keyframes:
        g = db.gt_future(tok, 4.0, 0.5)
        if not g['valid'].all():
            continue
        d = db.direction(tok)
        if d == 'FWD' and g['xy'][-1, 0] > 6:
            (turn if abs(g['xy'][-1, 1]) > 2.0 else fwd).append(tok)
        elif d == 'REV' and g['xy'][-1, 0] < -2:
            rev.append(tok)
    sel = [fwd[i] for i in np.linspace(0, len(fwd) - 1, n_fwd).astype(int)] if fwd else []
    sel += turn[len(turn) // 2: len(turn) // 2 + 1] + turn[:1] + rev[len(rev) // 2: len(rev) // 2 + 1]
    return list(dict.fromkeys(sel))


def stitched(inp, tok):
    import cv2
    l0 = inp.camera_image(tok, 'CAM_L0')[28:-28, 416:-416]
    f0 = inp.camera_image(tok, 'CAM_F0')[28:-28]
    r0 = inp.camera_image(tok, 'CAM_R0')[28:-28, 416:-416]
    return cv2.resize(np.concatenate([l0, f0, r0], 1), (1024, 256))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', nargs='+', required=True)
    ap.add_argument('--tokens', nargs='*', default=None)
    ap.add_argument('--aux', default=None, help='npz from run_agent.py --dump-aux (TransFuser-style outputs)')
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    ap.add_argument('--out', default=os.path.join(ROOT, 'outputs', 'vis', 'navsim'))
    ap.add_argument('--print-tokens', action='store_true')
    ap.add_argument('--top1', action='store_true', help='draw only trajs[0] of every variant')
    ap.add_argument('--prefix', default='', help='output file name prefix')
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    db = ParkingDataset(args.dataroot, args.scenes, load_agents=False)
    toks = args.tokens or pick_tokens(db)
    if args.print_tokens:
        print(' '.join(toks))
        return
    inp = OLPNavsimInputs(db)
    preds = []
    for p in args.pred:
        d = json.load(open(p))
        preds.append((d['method'], d['predictions']))
    aux = np.load(args.aux) if args.aux else None
    os.makedirs(args.out, exist_ok=True)
    for tok in toks:
        fig = plt.figure(figsize=(12, 10.5))
        ax0 = fig.add_axes([0.02, 0.76, 0.96, 0.22])
        ax0.imshow(stitched(inp, tok))
        ax0.set_axis_off()
        ax0.set_title('%s  %s  dir=%s  cmd=%s  (virtual nuPlan l0|f0|r0 as fed to TransFuser)' %
                      (db.scene_name(tok), tok[:10], db.direction(tok), db.command(tok)), fontsize=10)
        ax = fig.add_axes([0.08, 0.04, 0.84, 0.68])
        pc = inp.lidar_pc(tok)
        m = (pc[2] > 0.2) & (pc[2] < 3.0) & (np.abs(pc[0]) < 32) & (np.abs(pc[1]) < 32)
        ax.scatter(-pc[1, m], pc[0, m], s=0.2, c='0.6', rasterized=True)
        ego = ego_box_corners([0, 0], 0.0)
        ax.fill(-ego[:, 1], ego[:, 0], color='k', alpha=0.3)
        g = db.gt_future(tok, 4.0, 0.5)
        ax.plot(-np.r_[0, g['xy'][:, 1]], np.r_[0, g['xy'][:, 0]], 'k-o', ms=3, lw=2, label='GT 0-4 s')
        for j, (name, P) in enumerate(preds):
            e = P.get(tok)
            if e is None:
                continue
            tr = np.array(e['trajs'])
            c = COLORS[j % len(COLORS)]
            for k in range(1, 1 if args.top1 else len(tr)):
                ax.plot(-np.r_[0, tr[k, :, 1]], np.r_[0, tr[k, :, 0]], '-', color=c, lw=0.5, alpha=0.4)
            ax.plot(-np.r_[0, tr[0, :, 1]], np.r_[0, tr[0, :, 0]], '-o', color=c, ms=3, lw=1.8,
                    label='%s (K=%d)' % (name, len(tr)))
        if aux is not None and ('%s__agent_states' % tok) in aux.files:
            st = aux['%s__agent_states' % tok]
            lab = 1 / (1 + np.exp(-aux['%s__agent_labels' % tok]))
            for s_, l_ in zip(st, lab):
                if l_ < 0.5:
                    continue
                cb = box_corners_2d(s_[0], s_[1], s_[3], s_[4], s_[2])
                ax.fill(-cb[:, 1], cb[:, 0], fill=False, ec='tab:orange', lw=1.2)
            ax.plot([], [], color='tab:orange', label='TransFuser detections (p>0.5)')
        ax.set_xlim(-24, 24)
        ax.set_ylim(-14, 34)
        ax.set_aspect('equal')
        ax.set_xlabel('-y (m, right)')
        ax.set_ylabel('x (m, forward)')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        p = os.path.join(args.out, '%s%s_%s.png' % (args.prefix, db.scene_name(tok), tok[:10]))
        fig.savefig(p, dpi=80)
        plt.close(fig)
        print('wrote', p)


if __name__ == '__main__':
    main()
