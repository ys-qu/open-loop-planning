"""Causal per-keyframe inputs shared by the VLM / VLA adapters (numpy only).

Everything here uses only ego poses at or before the keyframe, except
`ParkingDataset.command()` (the nuScenes-style command derived from the GT future, the one
privileged input allowed by the protocol).

Conventions: ego frame x forward, y left (metres), origin = ego origin (~rear axle) at the
keyframe. Angles in radians, counter-clockwise (left) positive.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OLP_DERIVED = os.environ.get('OLP_DERIVED', os.path.join(ROOT, 'derived'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from olpcommon import DEFAULT_DATAROOT, ParkingDataset, read_scene_list  # noqa: E402,F401
from olpcommon.geometry import rot2d  # noqa: E402

NUSC_VIRTUAL_INFOS = os.path.join(_OLP_DERIVED, 'nusc_virtual', 'infos_%s.json' % os.environ.get('OLP_TAG', 'subset'))
# virtual nuScenes LIDAR_TOP origin in the dataset ego frame (tools/make_nusc_virtual.py)
NUSC_LIDAR_OFFSET = (0.94, 0.0)
# Renault Zoe (nuScenes car) wheelbase, and the steering-wheel / road-wheel ratio fitted on
# Impromptu-VLA's nuscenes_test.json (CAN steering vs. path curvature, see impromptu README)
NUSC_WHEELBASE = 2.588
NUSC_STEER_RATIO = 15.0


# ---------------------------------------------------------------------------- frames
def select_tokens(db, every=1, offset=0, limit=None, tokens=None):
    """Deterministic keyframe subset: every `every`-th keyframe (time order), starting at
    `offset`; optionally an explicit list of tokens."""
    toks = list(tokens) if tokens else list(db.keyframes)
    toks = toks[offset::every]
    if limit:
        toks = toks[:limit]
    return toks


# ---------------------------------------------------------------------------- kinematics
def _fit_state(t, xy, yaw, valid, t_ref, window_s=0.6, acc_window_s=1.0):
    """Same estimator as ParkingDataset.ego_state but at an arbitrary reference time t_ref
    (all samples must satisfy t <= t_ref). Returns (vx, vy, ax, ay, yaw_rate) in the
    frame of the poses (the caller rotates them into the body frame at t_ref)."""
    m = valid & (t <= t_ref + 1e-6) & (t >= t_ref - acc_window_s - 1e-6)
    if m.sum() < 3:
        return None
    tv, xyv, yawv = t[m] - t_ref, xy[m], np.unwrap(yaw[m])
    w = tv >= -window_s - 1e-6
    A = np.stack([tv[w], np.ones(w.sum())], 1)
    v = np.linalg.lstsq(A, xyv[w], rcond=None)[0][0]
    yr = float(np.linalg.lstsq(A, yawv[w], rcond=None)[0][0])
    a = np.zeros(2)
    if len(tv) >= 5:
        A2 = np.stack([tv ** 2, tv, np.ones(len(tv))], 1)
        a = 2 * np.linalg.lstsq(A2, xyv, rcond=None)[0][0]
    return v, a, yr


def past_states(db, token, offsets_s, window_s=0.6, acc_window_s=1.0):
    """Ego poses + causal kinematic states at past times t0+off (off <= 0).

    Returns a list (same order as offsets) of dicts:
      xy [2], yaw: ego-origin pose in the CURRENT ego frame
      vx, vy, ax, ay: velocity / acceleration in the BODY frame at that time (m/s, m/s^2)
      yaw_rate (rad/s), speed (signed, negative when reversing), valid (bool)
    The state at off uses only poses in [off - acc_window_s, off]."""
    offsets_s = [float(o) for o in offsets_s]
    n = int(np.ceil((acc_window_s - min(offsets_s)) / 0.1 - 1e-6))
    grid = np.round(-np.arange(n, -1, -1) * 0.1, 3)  # 0.1 s grid ending at 0 (as ego_state)
    grid = np.unique(np.concatenate([grid, np.round(offsets_s, 3)]))
    xy, yaw, valid = db.relative_poses(token, grid)
    out = []
    for off in offsets_s:
        k = int(np.argmin(np.abs(grid - off)))
        d = {'t': off, 'xy': xy[k].copy(), 'yaw': float(yaw[k]), 'valid': bool(valid[k]),
             'vx': 0.0, 'vy': 0.0, 'ax': 0.0, 'ay': 0.0, 'yaw_rate': 0.0, 'speed': 0.0}
        if valid[k]:
            r = _fit_state(grid, xy, yaw, valid, off, window_s, acc_window_s)
            if r is not None:
                v, a, yr = r
                R = rot2d(-yaw[k])  # current-frame vectors -> body frame at `off`
                vb, ab = R @ v, R @ a
                d.update(vx=float(vb[0]), vy=float(vb[1]), ax=float(ab[0]), ay=float(ab[1]),
                         yaw_rate=yr, speed=float(np.sign(vb[0]) * np.linalg.norm(vb)))
        out.append(d)
    return out


def curvature_from_state(st, min_speed=0.3):
    """Path curvature (1/m, left positive, steering-consistent sign when reversing):
    yaw_rate / signed speed; 0 when |speed| < min_speed."""
    v = st['speed']
    if abs(v) < min_speed:
        return 0.0
    return float(st['yaw_rate'] / v)


def nusc_steering_from_state(st, min_speed=0.3, clip=7.5):
    """Proxy for the nuScenes CAN 'steeranglefeedback' (steering-wheel angle, rad, left +):
    ratio * atan(wheelbase_zoe * curvature). The dataset has no steering signal."""
    k = curvature_from_state(st, min_speed)
    return float(np.clip(NUSC_STEER_RATIO * np.arctan(NUSC_WHEELBASE * k), -clip, clip))


def point_positions(states, offset):
    """Positions (current ego frame) of a vehicle-fixed point at `offset`=(dx, dy) given ego
    poses (list of past_states dicts). Returns [N, 2]."""
    off = np.asarray(offset, dtype=np.float64)
    return np.array([s['xy'] + rot2d(s['yaw']) @ off for s in states])


# ---------------------------------------------------------------------------- outputs
def const_vel_fallback(db, token, t):
    """Constant-velocity trajectory (ego frame) used when a VLM output cannot be parsed."""
    s = db.ego_state(token)
    v = np.array([s['vx'], s['vy']])
    t = np.asarray(t, dtype=np.float64)
    return v[None] * t[:, None]


def integrate_speed_curvature(speeds, curvs, dt=0.5, v0=None, k0=None):
    """Integrate future (speed, curvature) samples at t = dt, 2dt, ... into ego-frame positions.

    Trapezoidal rule from the current state (v0, k0) at t=0 (defaults: first sample).
    Signed speed: negative = reversing (moves backwards along the heading)."""
    v = np.asarray(speeds, dtype=np.float64)
    k = np.asarray(curvs, dtype=np.float64)
    v0 = v[0] if v0 is None else float(v0)
    k0 = k[0] if k0 is None else float(k0)
    vv = np.concatenate([[v0], v])
    kk = np.concatenate([[k0], k])
    yawrate = vv * kk
    th = np.concatenate([[0.0], np.cumsum(0.5 * (yawrate[1:] + yawrate[:-1]) * dt)])
    vx, vy = vv * np.cos(th), vv * np.sin(th)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * dt)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * dt)])
    return np.stack([x[1:], y[1:]], 1)


PARKING_PATTERNS = {
    'park*': r'\bpark(?:ed|ing|s)?\b',
    'stall': r'\bstalls?\b',
    'spot': r'\bspots?\b',
    'space': r'\b(?:parking )?spaces?\b',
    'revers*': r'\brevers(?:e|ed|ing)\b',
    'back up/out': r'\bback(?:s|ed|ing)? (?:up|out|in(?:to)?)\b',
    'pull in/out': r'\bpull(?:s|ed|ing)? (?:in(?:to)?|out)\b',
    'garage': r'\bgarages?\b',
    'parking lot': r'\bparking lots?\b',
    'maneuver': r'\bmaneuv\w*|\bmanoeuv\w*',
    'aisle': r'\baisles?\b',
}
# on-road ("highway-centric") vocabulary, for contrast (cf. the Alpamayo CoC analysis in the paper)
ONROAD_PATTERNS = {
    'lane': r'\blanes?\b',
    'keep lane': r'\bkeep(?:s|ing)? (?:in )?(?:the |its |my )?lane\b',
    'traffic light': r'\btraffic (?:light|signal)s?\b',
    'stop sign': r'\bstop signs?\b',
    'intersection': r'\bintersections?\b',
    'lane change': r'\b(?:change|changing) lanes?\b|\blane change',
    'speed bump': r'\bspeed bumps?\b',
    'crosswalk': r'\bcrosswalks?\b|\bzebra\b',
    'highway': r'\bhighways?\b|\bfreeway\b',
}


def parking_vocab(text):
    """Which parking-related terms (regex word matches) occur in a text."""
    import re
    t = (text or '').lower()
    return sorted(k for k, p in PARKING_PATTERNS.items() if re.search(p, t))


def onroad_vocab(text):
    """Which on-road / highway-centric terms occur in a text."""
    import re
    t = (text or '').lower()
    return sorted(k for k, p in ONROAD_PATTERNS.items() if re.search(p, t))
