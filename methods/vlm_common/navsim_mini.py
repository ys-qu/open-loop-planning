"""Real NAVSIM (OpenScene mini split) samples in the same structure the NAVSIM-VLA adapters
build for the parking dataset, used ONLY to validate the adapters (prompt / preprocessing / output
conventions) on in-domain data before the zero-shot runs on the parking dataset.

Sample dict: token, log, cams {CAM_F0/L0/R0: [4 paths, oldest first]}, hist [4, 3]
(x, y, heading of frames t-1.5..t0 in the current ego frame), vel [2], acc [2],
command one-hot [4], gt [N, 3] future poses (0.5 s steps) in the current ego frame.
"""
import glob
import os
import pickle

import numpy as np

NAVSIM_ROOT = os.environ.get('NAVSIM_MINI_ROOT', '')  # local NAVSIM mini split (optional positive check)
_ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'),
            ('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar'),
            ('numpy._core.multiarray', 'scalar'), ('collections', 'OrderedDict'),
            ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer')}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError('blocked %s.%s' % (module, name))


def _pose2d(T):
    return T[:2, 3].copy(), float(np.arctan2(T[1, 0], T[0, 0]))


def load_samples(n=40, n_future=8, seed=0, split='mini', min_speed=1.0, root=NAVSIM_ROOT):
    logs = sorted(glob.glob(os.path.join(root, 'navsim_logs', split, '*.pkl')))
    rng = np.random.RandomState(seed)
    cand = []
    for lp in logs:
        d = _NumpyOnlyUnpickler(open(lp, 'rb')).load()
        for i in range(3, len(d) - n_future, 10):
            win = d[i - 3:i + n_future + 1]
            if any(win[k + 1]['timestamp'] - win[k]['timestamp'] > 0.6e6 for k in range(len(win) - 1)):
                continue
            if np.hypot(*d[i]['ego_dynamic_state'][:2]) < min_speed:
                continue
            cand.append((lp, i, d))
    pick = rng.choice(len(cand), min(n, len(cand)), replace=False)
    out = []
    for j in sorted(pick):
        lp, i, d = cand[j]
        T0i = np.linalg.inv(np.asarray(d[i]['ego2global']))

        def rel(k):
            xy, yaw = _pose2d(T0i @ np.asarray(d[k]['ego2global']))
            return [xy[0], xy[1], yaw]
        f = d[i]
        sb = os.path.join(root, 'sensor_blobs', split)
        cams = {c: [os.path.join(sb, d[k]['cams'][c]['data_path']) for k in range(i - 3, i + 1)]
                for c in ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1')}
        out.append({
            'token': f['token'], 'log': f['log_name'], 'cams': cams,
            'hist': np.array([rel(k) for k in range(i - 3, i + 1)]),
            'vel': np.array(f['ego_dynamic_state'][:2], dtype=np.float64),
            'acc': np.array(f['ego_dynamic_state'][2:4], dtype=np.float64),
            'command': np.asarray(f['driving_command']).astype(np.int64),
            'gt': np.array([rel(k) for k in range(i + 1, i + 1 + n_future)]),
        })
    return out


def ade_fde(pred_xy, gt_xy, horizon_steps=None):
    n = min(len(pred_xy), len(gt_xy)) if horizon_steps is None else horizon_steps
    e = np.linalg.norm(np.asarray(pred_xy)[:n] - np.asarray(gt_xy)[:n, :2], axis=1)
    return float(e.mean()), float(e[-1])
