#!/usr/bin/env python
"""Visual sanity check: front image (CAM_5) with GT / predicted trajectories projected
on the ground plane, plus a BEV panel (history, GT future, all K predictions).

  python methods/vlm_common/viz.py --pred outputs/<variant>/predictions.json --n 8 \
      --out outputs/<variant>/viz
Picks frames spread over FWD / STP / REV unless --tokens is given.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from olp_inputs import DEFAULT_DATAROOT, ROOT, ParkingDataset  # noqa: E402
from olpcommon import GROUND_Z_EGO  # noqa: E402
from olpcommon.geometry import inv_T  # noqa: E402


def project(cam, xy):
    pts = np.c_[xy, np.full(len(xy), GROUND_Z_EGO), np.ones(len(xy))]
    pc = (inv_T(cam['sensor2ego']) @ pts.T)[:3]
    ok = pc[2] > 0.5
    uv = (cam['K'] @ pc)[:2] / np.maximum(pc[2], 1e-6)
    return uv.T, ok


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    ap.add_argument('--tokens', default=None)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    pred = json.load(open(args.pred))
    out = args.out or os.path.join(os.path.dirname(args.pred), 'viz')
    os.makedirs(out, exist_ok=True)
    db = ParkingDataset(args.dataroot, args.scenes, load_agents=False, verbose=False)
    P = pred['predictions']
    if args.tokens:
        toks = args.tokens.split(',')
    else:
        rng = np.random.RandomState(0)
        toks = []
        for d, n in (('FWD', args.n // 2), ('REV', args.n // 4), ('STP', args.n - args.n // 2 - args.n // 4)):
            c = [t for t in db.keyframes if t in P and db.direction(t) == d]
            toks += list(rng.choice(c, min(n, len(c)), replace=False)) if c else []
    for tok in toks:
        e = P[tok]
        t = np.asarray(e['t'])
        trajs = np.asarray(e['trajs'])
        g = db.gt_future(tok, max(3.0, float(t[-1])), 0.1)
        h = db.history(tok, 6, 0.5)
        cam = db.camera(tok, 'CAM_FRONT')
        fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={'width_ratios': [1.55, 1]})
        im = Image.open(cam['path']).convert('RGB')
        ax[0].imshow(im)
        for xy, c, lab in ((g['xy'][g['valid']], 'lime', 'GT'), (trajs[0], 'red', 'pred top-1')):
            uv, ok = project(cam, np.vstack([[0, 0], xy]))
            ax[0].plot(uv[ok, 0], uv[ok, 1], '-o', color=c, ms=3, lw=2, label=lab)
        ax[0].set_xlim(0, im.size[0]); ax[0].set_ylim(im.size[1], 0); ax[0].axis('off')
        ax[0].legend(loc='upper right')
        ex = e.get('extra', {})
        txt = (ex.get('text') or ex.get('motion_text') or '')
        ax[0].set_title('%s %s dir=%s cmd=%s %s' % (db.scene_name(tok), tok[:8], db.direction(tok),
                        db.command(tok), 'PARSE-FAIL' if ex.get('parse_fail') else ''), fontsize=9)
        if txt:
            fig.text(0.01, 0.01, txt[-600:].replace('\n', ' '), fontsize=6, wrap=True)
        a = ax[1]
        # BEV: x forward up, y left -> plot (-y, x)
        hv = h['xy'][h['valid']]
        a.plot(-hv[:, 1], hv[:, 0], 'k.-', label='history')
        gv = g['xy'][g['valid']]
        a.plot(-gv[:, 1], gv[:, 0], 'g-', lw=3, label='GT')
        for k in range(len(trajs) - 1, -1, -1):
            a.plot(-trajs[k][:, 1], trajs[k][:, 0], '-o', ms=3, color='red' if k == 0 else 'salmon',
                   lw=2 if k == 0 else 1, alpha=1 if k == 0 else 0.5, label='pred top-1' if k == 0 else None)
        a.plot(0, 0, 'b^', ms=10)
        a.set_aspect('equal'); a.grid(alpha=0.3)
        lim = max(8.0, np.abs(np.vstack([gv, trajs.reshape(-1, 2), hv])).max() + 1)
        a.set_xlim(-lim, lim); a.set_ylim(-lim * 0.6, lim * 1.2)
        a.set_xlabel('-y (right) [m]'); a.set_ylabel('x (forward) [m]'); a.legend(fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        p = os.path.join(out, '%s_%s_%s.jpg' % (db.direction(tok), db.scene_name(tok), tok[:8]))
        fig.savefig(p, dpi=70)
        plt.close(fig)
        print(p)


if __name__ == '__main__':
    main()
