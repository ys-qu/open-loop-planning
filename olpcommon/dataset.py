"""Keyframe loader of the parking dataset, shared by all method adapters.

numpy-only and python>=3.7 compatible, so it can be imported from every
method's conda env (old mmcv envs included).

Conventions (verified on the 24-scene subset, see docs/notes.md):
  * ego frame: x forward, y left, z up. Origin ~ rear axle; ground is at
    z ~= -0.45 m in this frame (LiDAR is ~2.18 m above ground).
  * calibrated_sensor.json extrinsics are sensor->ego in this frame.
  * agent boxes (labels/agents_bev_lc_refined_human_corr) are in the ego frame
    of their own keyframe, heading measured from +x.
  * ego_pose.json rotations are rotated 180 deg about z w.r.t. the ego frame
    (their +x points to the rear). ``ego2global`` below applies the fix:
        T_world_from_ego = T_pose @ diag(-1, -1, 1)
"""
import bisect
import json
import os

import numpy as np

from .geometry import (FLIP_Z180, inv_T, make_T, quat_to_rot, rot2d, rot_to_quat,
                       slerp, wrap_angle, yaw_of)

# Dataset camera -> nuScenes camera name (by mounting direction)
SRC_TO_NUSC_CAM = {
    'CAM_5': 'CAM_FRONT',
    'CAM_1': 'CAM_FRONT_LEFT',
    'CAM_2': 'CAM_FRONT_RIGHT',
    'CAM_4': 'CAM_BACK_LEFT',
    'CAM_3': 'CAM_BACK_RIGHT',
    'CAM_6': 'CAM_BACK',
}
NUSC_TO_SRC_CAM = {v: k for k, v in SRC_TO_NUSC_CAM.items()}
NUSC_CAM_ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                  'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
GROUND_Z_EGO = -0.45
POSE_SHIFT_US = -50000
# Ego vehicle footprint (m), and where its centre sits relative to the ego origin.
EGO_LENGTH, EGO_WIDTH = 4.94, 2.00
EGO_CENTER_X = 1.40


def read_scene_list(path_or_names):
    if path_or_names is None:
        return None
    if isinstance(path_or_names, (list, tuple)):
        return list(path_or_names)
    if os.path.isfile(path_or_names):
        with open(path_or_names) as f:
            return [l.strip() for l in f if l.strip() and not l.startswith('#')]
    return [s.strip() for s in path_or_names.split(',') if s.strip()]


class ParkingDataset(object):
    """Index over the 2 Hz keyframes of a list of scenes.

    Args:
        dataroot: unzipped dataset root (contains v1.0-trainval/, samples/, labels/...).
        scenes: list of scene names, a comma separated string, a txt file, or
            None for every scene that has sensor data on disk.
        load_agents: load the human-corrected agent boxes (needed for collision checks).
    """

    def __init__(self, dataroot, scenes=None, version='v1.0-trainval', load_agents=True,
                 verbose=True, pose_mode='dedup'):
        if not dataroot or not os.path.isdir(os.path.join(dataroot, version)):
            raise ValueError("Dataset root not found: %r. Set OLP_DATAROOT (source env.sh) "
                             "or pass --dataroot." % dataroot)
        self.root = dataroot
        tdir = os.path.join(dataroot, version)
        load = lambda n: json.load(open(os.path.join(tdir, n + '.json')))
        self.scene_by_name = {s['name']: s for s in load('scene')}
        self.scene_by_token = {s['token']: s for s in self.scene_by_name.values()}
        self.sample = {s['token']: s for s in load('sample')}
        self.calib = {c['token']: c for c in load('calibrated_sensor')}
        sensors = {s['token']: s for s in load('sensor')}
        self.calib_by_channel = {sensors[c['sensor_token']]['channel']: c for c in self.calib.values()}

        # ego poses (whole recording) for interpolation. ego_pose.json is nominally 10 Hz
        # but only changes every ~200 ms (41% of consecutive entries are exact duplicates).
        # pose_mode='dedup' keeps the first entry of each run of duplicates, shifted by
        # POSE_SHIFT_US (best match to LiDAR ICP odometry); 'raw' uses the table as is.
        ep = sorted(load('ego_pose'), key=lambda e: e['timestamp'])
        ts = np.array([e['timestamp'] for e in ep], dtype=np.int64)
        tr = np.array([e['translation'] for e in ep], dtype=np.float64)
        qs = np.array([rot_to_quat(quat_to_rot(e['rotation']) @ FLIP_Z180) for e in ep])
        self.pose_mode = pose_mode
        if pose_mode == 'dedup':
            keep = np.concatenate([[True], np.any(np.abs(np.diff(tr, axis=0)) > 1e-9, axis=1) |
                                   np.any(np.abs(np.diff(qs, axis=0)) > 1e-9, axis=1)])
            keep[-1] = True
            ts = ts[keep] + POSE_SHIFT_US
            tr, qs = tr[keep], qs[keep]
        elif pose_mode != 'raw':
            raise ValueError(pose_mode)
        self.pose_ts, self.pose_t, self.pose_q = ts, tr, qs

        names = read_scene_list(scenes)
        if names is None:
            names = sorted(n for n in self.scene_by_name)
        missing = [n for n in names if n not in self.scene_by_name]
        if missing:
            raise ValueError('unknown scenes: %s' % missing)
        self.scene_names = names
        scene_tokens = {self.scene_by_name[n]['token'] for n in names}
        self.keyframes = sorted([t for t, s in self.sample.items() if s['scene_token'] in scene_tokens],
                                key=lambda t: self.sample[t]['timestamp'])

        # sample_data: keyframe records per sample + full per-channel time series
        sd = load('sample_data')
        kf = set(self.keyframes)
        self.sd_by_sample = {}
        self.sd_series = {}
        t_lo = self.sample[self.keyframes[0]]['timestamp'] - int(3e6) if self.keyframes else 0
        t_hi = self.sample[self.keyframes[-1]]['timestamp'] + int(7e6) if self.keyframes else 0
        for x in sd:
            if x['sample_token'] in kf and x['is_key_frame']:
                self.sd_by_sample.setdefault(x['sample_token'], {})[x['channel']] = x
            if t_lo <= x['timestamp'] <= t_hi:
                self.sd_series.setdefault(x['channel'], []).append(x)
        for ch in self.sd_series:
            self.sd_series[ch].sort(key=lambda x: x['timestamp'])
        self._series_ts = {ch: [x['timestamp'] for x in v] for ch, v in self.sd_series.items()}

        # ego trajectory labels (direction FWD/STP/REV, speed)
        p = os.path.join(dataroot, 'labels', 'ego_trajectory.json')
        self.ego_label = {x['sample_token']: x for x in json.load(open(p))} if os.path.exists(p) else {}

        # Each scene file holds its 40 core keyframes plus 2 s / 6 s of context keyframes that
        # belong to the neighbouring scenes. Only a keyframe's own (core) copy is the verified
        # annotation; context copies are partial, so they are used only as a fallback.
        self.agents = {}
        if load_agents:
            adir = os.path.join(dataroot, 'labels', 'agents_bev_lc_refined_human_corr')
            fallback = {}
            for n in names:
                stok = self.scene_by_name[n]['token']
                f = os.path.join(adir, '%s_%s.json' % (n, stok))
                if os.path.exists(f):
                    for tok, lst in json.load(open(f)).items():
                        if tok in self.sample and self.sample[tok]['scene_token'] == stok:
                            self.agents[tok] = lst
                        else:
                            fallback.setdefault(tok, lst)
            for tok, lst in fallback.items():
                self.agents.setdefault(tok, lst)
        if verbose:
            print('[ParkingDataset] %d scenes, %d keyframes, agents for %d keyframes' %
                  (len(names), len(self.keyframes), sum(1 for t in self.keyframes if t in self.agents)))

    # ------------------------------------------------------------------ basics
    def scene_name(self, token):
        return self.scene_by_token[self.sample[token]['scene_token']]['name']

    def timestamp(self, token):
        return self.sample[token]['timestamp']

    def prev_keyframe(self, token):
        """Previous 2 Hz keyframe in time (may belong to the previous scene), or None."""
        s = self.sample[token]
        if s['prev']:
            return s['prev']
        i = self.keyframes.index(token)
        if i > 0 and self.timestamp(token) - self.timestamp(self.keyframes[i - 1]) < 0.75e6:
            return self.keyframes[i - 1]
        return None

    def next_keyframe(self, token):
        s = self.sample[token]
        if s['next']:
            return s['next']
        i = self.keyframes.index(token)
        if i + 1 < len(self.keyframes) and self.timestamp(self.keyframes[i + 1]) - self.timestamp(token) < 0.75e6:
            return self.keyframes[i + 1]
        return None

    # ------------------------------------------------------------------ poses
    def ego2global_at(self, ts):
        """4x4 T_world_from_ego (corrected) at an arbitrary timestamp (us); None if out of range."""
        if ts < self.pose_ts[0] or ts > self.pose_ts[-1]:
            return None
        i = int(np.searchsorted(self.pose_ts, ts))
        if i == 0 or self.pose_ts[i] == ts:
            return make_T(quat_to_rot(self.pose_q[i]), self.pose_t[i])
        t0, t1 = self.pose_ts[i - 1], self.pose_ts[i]
        a = float(ts - t0) / float(t1 - t0)
        q = slerp(self.pose_q[i - 1], self.pose_q[i], a)
        return make_T(quat_to_rot(q), (1 - a) * self.pose_t[i - 1] + a * self.pose_t[i])

    def ego2global(self, token):
        """T_world_from_ego at the keyframe (LiDAR keyframe timestamp)."""
        return self.ego2global_at(self.timestamp(token))

    def sd_ego2global(self, sd_rec):
        """T_world_from_ego at a sample_data record's own timestamp."""
        return self.ego2global_at(sd_rec['timestamp'])

    def relative_poses(self, token, offsets_s):
        """Ego poses at t0+offset expressed in the ego frame at t0.

        Returns (xy [N,2], yaw [N], valid [N] bool)."""
        T0i = inv_T(self.ego2global(token))
        ts0 = self.timestamp(token)
        xy = np.zeros((len(offsets_s), 2))
        yaw = np.zeros(len(offsets_s))
        valid = np.zeros(len(offsets_s), dtype=bool)
        for k, off in enumerate(offsets_s):
            T = self.ego2global_at(ts0 + int(round(off * 1e6)))
            if T is None:
                continue
            Tr = T0i @ T
            xy[k] = Tr[:2, 3]
            yaw[k] = yaw_of(Tr[:3, :3])
            valid[k] = True
        return xy, yaw, valid

    def gt_future(self, token, horizon_s=6.4, dt=0.1):
        """Ground-truth future ego trajectory in the current ego frame.

        Returns dict(t [N], xy [N,2], yaw [N], valid [N])."""
        t = np.round(np.arange(1, int(round(horizon_s / dt)) + 1) * dt, 6)
        xy, yaw, valid = self.relative_poses(token, t)
        return {'t': t, 'xy': xy, 'yaw': yaw, 'valid': valid}

    def history(self, token, n=4, dt=0.5):
        """Past ego poses (oldest first, excluding t0) in the current ego frame."""
        t = -np.arange(n, 0, -1) * dt
        xy, yaw, valid = self.relative_poses(token, t)
        return {'t': t, 'xy': xy, 'yaw': yaw, 'valid': valid}

    def ego_state(self, token, window_s=0.6, acc_window_s=1.0):
        """Kinematic state at the keyframe, CAUSAL (only poses in [t0 - window, t0]).

        Velocity / yaw rate: least-squares line fit over the last `window_s` (sampled at
        0.1 s); acceleration: quadratic fit over the last `acc_window_s`.
        Returns dict with vx, vy (m/s, ego frame), ax, ay (m/s^2), yaw_rate (rad/s),
        speed (signed: negative when reversing)."""
        t = -np.arange(int(round(acc_window_s / 0.1)), -1, -1) * 0.1
        xy, yaw, valid = self.relative_poses(token, t)
        if valid.sum() < 3 or not valid[-1]:
            return dict(vx=0.0, vy=0.0, ax=0.0, ay=0.0, yaw_rate=0.0, speed=0.0)
        tv, xyv, yawv = t[valid], xy[valid], np.unwrap(yaw[valid])
        m = tv >= -window_s - 1e-6
        A = np.stack([tv[m], np.ones(m.sum())], 1)
        v = np.linalg.lstsq(A, xyv[m], rcond=None)[0][0]
        w = float(np.linalg.lstsq(A, yawv[m], rcond=None)[0][0])
        a = np.zeros(2)
        if len(tv) >= 5:
            A2 = np.stack([tv ** 2, tv, np.ones(len(tv))], 1)
            a = 2 * np.linalg.lstsq(A2, xyv, rcond=None)[0][0]
        return dict(vx=float(v[0]), vy=float(v[1]), ax=float(a[0]), ay=float(a[1]),
                    yaw_rate=w, speed=float(np.sign(v[0]) * np.linalg.norm(v)))

    def command(self, token, horizon_s=3.0, lat_thresh=2.0):
        """nuScenes/VAD-style high-level command derived from the GT future.

        Returns 'LEFT', 'RIGHT' or 'STRAIGHT' (lateral offset of the GT position at
        horizon_s, or of the last valid one)."""
        g = self.gt_future(token, horizon_s, 0.5)
        if not g['valid'].any():
            return 'STRAIGHT'
        y = g['xy'][np.where(g['valid'])[0][-1], 1]
        if y >= lat_thresh:
            return 'LEFT'
        if y <= -lat_thresh:
            return 'RIGHT'
        return 'STRAIGHT'

    def direction(self, token):
        """Ego motion label from labels/ego_trajectory.json: FWD / STP / REV."""
        return self.ego_label.get(token, {}).get('direction')

    # ------------------------------------------------------------------ sensors
    def camera(self, token, channel):
        """Keyframe camera record. channel: dataset name (CAM_1..6) or nuScenes name.

        Returns dict(path, K [3x3], sensor2ego [4x4], ego2global [4x4], timestamp, width, height)."""
        ch = NUSC_TO_SRC_CAM.get(channel, channel)
        rec = self.sd_by_sample[token][ch]
        return self._sd_info(rec)

    def cameras(self, token, nusc_order=True):
        names = NUSC_CAM_ORDER if nusc_order else ['CAM_%d' % i for i in range(1, 7)]
        return {n: self.camera(token, n) for n in names}

    def lidar(self, token):
        return self._sd_info(self.sd_by_sample[token]['LIDAR_TOP'])

    def load_lidar_points(self, path):
        """Nx5 float32 (x, y, z, intensity, ring) in the LiDAR frame."""
        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)

    def sweeps(self, token, channel, n_prev, include_current=False):
        """Previous non-key sample_data (10 Hz) for a channel, newest first."""
        ch = NUSC_TO_SRC_CAM.get(channel, channel)
        rec = self.sd_by_sample[token][ch]
        ts_list = self._series_ts[ch]
        i = bisect.bisect_left(ts_list, rec['timestamp'])
        out = []
        start = i if include_current else i - 1
        for j in range(start, max(-1, start - n_prev), -1):
            r = self.sd_series[ch][j]
            p = os.path.join(self.root, r['filename'])
            if os.path.exists(p):
                out.append(self._sd_info(r))
        return out

    def _sd_info(self, rec):
        c = self.calib[rec['calibrated_sensor_token']]
        info = {
            'path': os.path.join(self.root, rec['filename']),
            'channel': rec['channel'],
            'timestamp': rec['timestamp'],
            'sensor2ego': make_T(quat_to_rot(c['rotation']), c['translation']),
            'ego2global': self.sd_ego2global(rec),
            'width': rec.get('width', 0),
            'height': rec.get('height', 0),
            'token': rec['token'],
        }
        if c.get('camera_intrinsic'):
            info['K'] = np.array(c['camera_intrinsic'], dtype=np.float64)
        return info

    # ------------------------------------------------------------------ agents
    def agent_boxes(self, token, ref_token=None):
        """Agent boxes of keyframe `token`, optionally expressed in the ego frame of `ref_token`.

        Returns list of dicts: track_id, cls, xy [2], yaw, length, width, is_static."""
        out = []
        lst = self.agents.get(token, [])
        if ref_token is not None and ref_token != token:
            Tr = inv_T(self.ego2global(ref_token)) @ self.ego2global(token)
            R2, t2, dyaw = Tr[:2, :2], Tr[:2, 3], yaw_of(Tr[:3, :3])
        else:
            R2, t2, dyaw = np.eye(2), np.zeros(2), 0.0
        for a in lst:
            xy = R2 @ np.array(a['bev_center'], dtype=np.float64) + t2
            out.append({'track_id': a['track_id'], 'cls': a['cls'], 'xy': xy,
                        'yaw': float(wrap_angle(a['bev_heading'] + dyaw)),
                        'length': float(a['bev_size'][0]), 'width': float(a['bev_size'][1]),
                        'is_static': bool(a.get('is_static', False))})
        return out

    def has_agents(self, token):
        return token in self.agents


def ego_box_corners(xy, yaw):
    """Corners of the ego footprint when the ego reference point is at xy with heading yaw."""
    from .geometry import box_corners_2d
    c = np.asarray(xy) + rot2d(yaw) @ np.array([EGO_CENTER_X, 0.0])
    return box_corners_2d(c[0], c[1], EGO_LENGTH, EGO_WIDTH, yaw)
