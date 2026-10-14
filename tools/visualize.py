#!/usr/bin/env python
"""Qualitative check: front camera + BEV with GT future, labeled agents and the plans of
one or more methods (optionally the first method's own detections / map from `extra`).

  python tools/visualize.py --pred outputs/vad_base/predictions.json outputs/const_vel/predictions.json \
      --tokens auto --n 6 --out outputs/vis/vad_vs_cv
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olpcommon import DEFAULT_DATAROOT, ParkingDataset, ego_box_corners, load_predictions  # noqa: E402
from olpcommon.geometry import box_corners_2d  # noqa: E402

COLORS = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22']


def pick_tokens(db, n, seed=0):
    """A spread of interesting frames: moving forward, reversing, and stopped."""
    rng = np.random.RandomState(seed)
    out = []
    for d, k in (('FWD', (n + 1) // 2), ('REV', n // 3), ('STP', n - (n + 1) // 2 - n // 3)):
        c = [t for t in db.keyframes if db.direction(t) == d]
        if c and k > 0:
            out += list(rng.choice(c, min(k, len(c)), replace=False))
    return out


def draw(db, tok, preds, names, out_png, show_extra=True, rng_m=30):
    fig = plt.figure(figsize=(15, 6.2))
    ax0 = fig.add_axes([0.0, 0.05, 0.55, 0.9])
    cam = db.camera(tok, 'CAM_FRONT')
    ax0.imshow(Image.open(cam['path']).resize((1440, 930)))
    ax0.set_axis_off()
    ax0.set_title('%s  %s  dir=%s  cmd=%s' % (db.scene_name(tok), tok[:8], db.direction(tok), db.command(tok)),
                  fontsize=10)
    ax = fig.add_axes([0.58, 0.07, 0.4, 0.86])
    # labeled agents (current keyframe)
    for a in db.agent_boxes(tok):
        c = box_corners_2d(a['xy'][0], a['xy'][1], a['length'], a['width'], a['yaw'])
        col = '#7f7f7f' if a['cls'] == 'car' else '#2ca02c'
        ax.fill(c[:, 1] * -1, c[:, 0], facecolor='none', edgecolor=col, lw=0.8)
    e = ego_box_corners([0, 0], 0.0)
    ax.fill(-e[:, 1], e[:, 0], color='k', alpha=0.6)
    # model's own perception from the first prediction file
    if show_extra and preds:
        ex = preds[0]['predictions'].get(tok, {}).get('extra', {})
        for d in ex.get('dets_ego[x,y,l,w,yaw,score,label]', []):
            c = box_corners_2d(d[0], d[1], d[2], d[3], d[4])  # [x, y, length, width, yaw] in the ego frame
            ax.plot(np.r_[-c[:, 1], -c[0, 1]], np.r_[c[:, 0], c[0, 0]], color=COLORS[0], lw=0.6, ls=':')
        for m in ex.get('map_ego', []):
            p = np.array(m['pts'])
            ax.plot(-p[:, 1], p[:, 0], color={'divider': 'orange', 'ped_crossing': 'cyan', 'boundary': 'purple'}.get(
                m['cls'], 'k'), lw=1.0, alpha=0.7)
    g = db.gt_future(tok, 6.4, 0.1)
    v = g['valid']
    ax.plot(-g['xy'][v, 1], g['xy'][v, 0], color='#2ca02c', lw=5, alpha=0.6, zorder=10, label='GT 6.4 s')
    for i, (p, nm) in enumerate(zip(preds, names)):
        e = p['predictions'].get(tok)
        if e is None:
            continue
        for k, tr in enumerate(e['trajs']):
            ax.plot(-tr[:, 1], tr[:, 0], color=COLORS[i % len(COLORS)], lw=2 if k == 0 else 0.7,
                    alpha=1 if k == 0 else 0.5, marker='o' if k == 0 else None, ms=2.5,
                    label=nm if k == 0 else None)
    ax.set_xlim(-rng_m / 2, rng_m / 2)
    ax.set_ylim(-rng_m / 3, rng_m)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('-y (right) [m]')
    ax.set_ylabel('x (forward) [m]')
    ax.legend(fontsize=8, loc='upper left')
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', nargs='+', required=True)
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scenes_subset.txt')))
    ap.add_argument('--tokens', default='auto', help="'auto' or comma separated sample tokens")
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--out', default='outputs/vis')
    ap.add_argument('--range', type=float, default=30)
    args = ap.parse_args()
    files = [f for p in args.pred for f in sorted(glob.glob(p))]
    preds = [load_predictions(f) for f in files]
    names = [p.get('method', os.path.basename(os.path.dirname(f))) for p, f in zip(preds, files)]
    db = ParkingDataset(args.dataroot, args.scenes, verbose=False)
    if args.tokens == 'auto':
        toks = pick_tokens(db, args.n)
    else:  # full tokens or unique prefixes
        toks = [next(k for k in db.keyframes if k.startswith(p)) for p in args.tokens.split(',')]
    os.makedirs(args.out, exist_ok=True)
    for t in toks:
        p = os.path.join(args.out, '%s_%s_%s.png' % (db.scene_name(t), db.direction(t), t[:8]))
        draw(db, t, preds, names, p, rng_m=args.range)
        print(p)


if __name__ == '__main__':
    main()
