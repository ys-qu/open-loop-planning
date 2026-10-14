#!/usr/bin/env python
"""Sanity-check open-loop planning metrics on the 2 Hz keyframes of the parking dataset.

This is NOT the official benchmark protocol; it is a quick check that a method's
outputs are sensible and in the right frame.

Metrics (all at 10 Hz: predictions are linearly interpolated to 0.1 s steps,
with (0, 0) prepended at t=0):
  ADE@H / FDE@H        top-1 plan, over (0, H] / at H
  minADE@H / minFDE@H  best of the K plans (equals ADE/FDE when K=1)
  col@3s               % of frames where the top-1 ego footprint overlaps a labeled
                       agent box at a future keyframe (0.5..3 s), agents taken from
                       those future keyframes
  rev_recall           % of REV frames (ego reversing) whose top-1 plan moves backwards
  The same collision check on the GT trajectory is printed as a label-noise floor.

Usage:
  python tools/eval_planning.py --pred outputs/*/predictions.json --scenes scenes_subset.txt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olpcommon import DEFAULT_DATAROOT, ParkingDataset, ego_box_corners, load_predictions  # noqa: E402
from olpcommon.geometry import box_corners_2d, headings_from_xy, polygons_intersect  # noqa: E402

TRAVEL_BUCKETS = [('stationary(<1m)', 0, 1), ('crawl(1-5m)', 1, 5), ('moving(5-15m)', 5, 15), ('fast(>15m)', 15, 1e9)]


def interp_traj(t_src, xy_src, t_dst):
    t = np.concatenate([[0.0], t_src])
    xy = np.vstack([[0.0, 0.0], xy_src])
    return np.stack([np.interp(t_dst, t, xy[:, 0]), np.interp(t_dst, t, xy[:, 1])], axis=1)


def collides(db, token, xy_fn, horizon=3.0):
    """xy_fn(t) -> (xy, yaw) of ego at time t. Checks future keyframes up to horizon."""
    if not db.has_agents(token):
        return None
    cur, t = token, 0.0
    t0 = db.timestamp(token)
    while True:
        cur = db.next_keyframe(cur)
        if cur is None:
            break
        t = (db.timestamp(cur) - t0) / 1e6
        if t > horizon + 0.05:
            break
        if not db.has_agents(cur):
            continue
        xy, yaw = xy_fn(t)
        ego = ego_box_corners(xy, yaw)
        for a in db.agent_boxes(cur, ref_token=token):
            if np.linalg.norm(a['xy'] - xy) > 12:
                continue
            if polygons_intersect(ego, box_corners_2d(a['xy'][0], a['xy'][1], a['length'], a['width'], a['yaw'])):
                return True
    return False


def evaluate(db, pred, horizons, dt=0.1):
    rows = []
    for tok in db.keyframes:
        e = pred['predictions'].get(tok)
        if e is None:
            continue
        native_h = float(e['t'][-1])
        hmax = max(horizons + [native_h])
        gt = db.gt_future(tok, hmax, dt)
        trajs = e['trajs']  # K x T x 2
        P = np.stack([interp_traj(e['t'], tr, gt['t']) for tr in trajs])  # K x N x 2
        err = np.linalg.norm(P - gt['xy'][None], axis=2)  # K x N
        r = {'token': tok, 'direction': db.direction(tok), 'native_h': native_h, 'K': len(trajs)}
        g64 = db.gt_future(tok, 6.4, dt)
        if g64['valid'].all():
            pts = np.vstack([[0, 0], g64['xy']])
            r['travel'] = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        for H in horizons + ['native']:
            h = native_h if H == 'native' else float(H)
            if h > native_h + 1e-6:
                continue
            m = gt['t'] <= h + 1e-6
            if not gt['valid'][m].all():
                continue
            ade = err[:, m].mean(axis=1)
            fde = err[:, m][:, -1]
            r['ADE@%s' % H], r['FDE@%s' % H] = float(ade[0]), float(fde[0])
            r['minADE@%s' % H], r['minFDE@%s' % H] = float(ade.min()), float(fde.min())
        # collisions (top-1, up to 3 s or native horizon)
        top = trajs[0]
        tt = np.concatenate([[0.0], e['t']])
        yaws = np.concatenate([[0.0], headings_from_xy(top)])
        xyp = np.vstack([[0.0, 0.0], top])

        def pred_pose(t):
            return (np.array([np.interp(t, tt, xyp[:, 0]), np.interp(t, tt, xyp[:, 1])]),
                    float(np.interp(t, tt, np.unwrap(yaws))))

        def gt_pose(t):
            g = db.relative_poses(tok, [t])
            return g[0][0], float(g[1][0])

        ch = min(3.0, native_h)
        r['col'] = collides(db, tok, pred_pose, ch)
        r['col_gt'] = collides(db, tok, gt_pose, ch)
        r['pred_backward'] = bool(np.interp(min(2.0, native_h), tt, xyp[:, 0]) < -0.3)
        rows.append(r)
    return rows


def summarize(rows, keys):
    out = {'n': len(rows)}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = (float(np.mean(v)), len(v)) if v else (float('nan'), 0)
    c = [r['col'] for r in rows if r.get('col') is not None]
    cg = [r['col_gt'] for r in rows if r.get('col_gt') is not None]
    out['col%'] = 100.0 * np.mean(c) if c else float('nan')
    out['col_gt%'] = 100.0 * np.mean(cg) if cg else float('nan')
    rev = [r['pred_backward'] for r in rows if r['direction'] == 'REV']
    out['rev_recall%'] = 100.0 * np.mean(rev) if rev else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', nargs='+', required=True, help='prediction json files (globs ok)')
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scenes_subset.txt')))
    ap.add_argument('--horizons', nargs='+', type=float, default=[1.0, 2.0, 3.0])
    ap.add_argument('--out', default=None, help='optional json summary path')
    ap.add_argument('--by', nargs='*', default=['direction', 'travel'], help='stratifications to print')
    args = ap.parse_args()

    files = sorted({f for p in args.pred for f in glob.glob(p)})
    db = ParkingDataset(args.dataroot, args.scenes)
    hs = [h if h != int(h) else int(h) for h in args.horizons]
    hs = [float(h) for h in hs]
    keys = []
    for H in [('%g' % h) for h in hs] + ['native']:
        keys += ['ADE@%s' % H, 'FDE@%s' % H, 'minADE@%s' % H, 'minFDE@%s' % H]
    allsum = {}
    for f in files:
        pred = load_predictions(f)
        rows = evaluate(db, pred, hs)
        # rename float horizons to %g keys
        for r in rows:
            for k in list(r.keys()):
                for h in hs:
                    for pre in ('ADE@', 'FDE@', 'minADE@', 'minFDE@'):
                        if k == '%s%s' % (pre, h):
                            r['%s%g' % (pre, h)] = r.pop(k)
        name = pred.get('method', os.path.basename(os.path.dirname(f)))
        s = {'all': summarize(rows, keys)}
        if 'direction' in args.by:
            for d in ('FWD', 'STP', 'REV'):
                s['dir=' + d] = summarize([r for r in rows if r['direction'] == d], keys)
        if 'travel' in args.by:
            for b, lo, hi in TRAVEL_BUCKETS:
                s['travel=' + b] = summarize([r for r in rows if 'travel' in r and lo <= r['travel'] < hi], keys)
        allsum[name] = s
        nat = rows[0]['native_h'] if rows else float('nan')
        Ks = sorted({r['K'] for r in rows})
        print('\n=== %s  (%s)  frames=%d  native horizon=%.1fs  K=%s' % (name, f, len(rows), nat, Ks))
        hdr = '%-24s %5s ' % ('subset', 'n') + ' '.join('%11s' % ('ADE/FDE@%g' % h) for h in hs) + \
              ' %13s %13s %7s %7s %8s' % ('ADE/FDE@nat', 'minADE@nat', 'col%', 'colGT%', 'revRec%')
        print(hdr)
        for sub, v in s.items():
            cells = []
            for h in hs:
                a, n = v['ADE@%g' % h]
                fd, _ = v['FDE@%g' % h]
                cells.append('%5.2f/%5.2f' % (a, fd) if n else '%11s' % '-')
            a, n = v['ADE@native']
            fd, _ = v['FDE@native']
            ma, _ = v['minADE@native']
            print('%-24s %5d ' % (sub, v['n']) + ' '.join(cells) +
                  ' %13s %13s %7.2f %7.2f %8.1f' % ('%5.2f/%5.2f' % (a, fd) if n else '-',
                                                     '%5.2f' % ma if n else '-',
                                                     v['col%'], v['col_gt%'], v['rev_recall%']))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(allsum, f, indent=1)
        print('summary written to', args.out)


if __name__ == '__main__':
    main()
