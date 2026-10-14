"""Build NAVSIM `AgentInput`s from the dataset keyframes (inference only).

Works with any NAVSIM fork on the python path (navsim v1.1 forks such as DiffusionDrive's, or
navsim v2.x): only the shared dataclasses AgentInput / EgoStatus / Cameras / Camera / Lidar
and SensorConfig.get_sensors_at_iteration are used.

Frames / conventions (checked on the local NAVSIM mini split, see methods/navsim/README.md):
  * NAVSIM ego frame = nuPlan rear-axle frame, x forward, y left, z up; the merged LiDAR
    point cloud is given in this frame (lidar2ego = identity) and the ground is at
    z = -0.33 m (rear-axle height), NOT at z = 0.
  * dataset ego frame: same axes, origin ~ rear axle, ground at z = -0.45 m.
    -> x, y are used unchanged; z is shifted by Z_SHIFT = (-0.33) - (-0.45) = +0.12 m for
       LiDAR points and camera positions.
  * History: NAVSIM agents get 4 frames at 2 Hz (t = -1.5, -1.0, -0.5, 0 s; index 3 =
    current). Dataset keyframes are 2 Hz too, so history frames = previous keyframes.
  * ego_status: velocity / acceleration (ego frame of that frame) from ParkingDataset.ego_state()
    (causal fits on the de-duplicated poses); ego_pose = (x, y, heading) relative to the
    current frame.
  * driving_command: one-hot [LEFT, STRAIGHT, RIGHT, UNKNOWN] (order verified on NAVSIM
    mini: index 0 frames turn left, 2 turn right) from ParkingDataset.command() (GT future,
    privileged). The current command is used for every history frame.
"""
import dataclasses
import os

import numpy as np

from olpcommon import EGO_CENTER_X, EGO_LENGTH, EGO_WIDTH, GROUND_Z_EGO
from olpcommon.geometry import inv_T

from .virtual_cam import NAVSIM_TO_SRC, VirtualCamRenderer, load_model
_OLP_DERIVED = os.environ.get('OLP_DERIVED', os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'), 'derived'))

NAVSIM_CMD_INDEX = {'LEFT': 0, 'STRAIGHT': 1, 'RIGHT': 2, 'UNKNOWN': 3}
CAM_ATTRS = ['cam_f0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0']
DEFAULT_VIRTUAL_DIR = os.path.join(_OLP_DERIVED, 'navsim_virtual')


def command_onehot(cmd):
    v = np.zeros(4, dtype=np.int64)
    v[NAVSIM_CMD_INDEX[cmd]] = 1
    return v


class OLPNavsimInputs(object):
    """dataset -> NAVSIM AgentInput.

    Args:
      db: olpcommon.ParkingDataset
      num_history: number of NAVSIM history frames (4 for all public NAVSIM agents)
      distort: virtual cameras reproduce the nuPlan lens distortion (True = like training)
      virtual_dir: pre-rendered virtual images (make_navsim_virtual.py); rendered on the
        fly if a file is missing
      lidar_sweeps: number of 10 Hz LiDAR sweeps merged into the keyframe cloud (1 = only
        the keyframe sweep; the dataset's 128-beam sweep is already denser than nuPlan's
        merged 5-LiDAR cloud)
      remove_ego_points: drop points inside the ego footprint (nuPlan clouds have none)
      command_fn: tok -> 'LEFT'|'STRAIGHT'|'RIGHT'|'UNKNOWN' (default ParkingDataset.command)
    """

    def __init__(self, db, num_history=4, distort=True, virtual_dir=DEFAULT_VIRTUAL_DIR,
                 lidar_sweeps=1, remove_ego_points=True, command_fn=None):
        self.db = db
        self.num_history = num_history
        self.distort = distort
        self.virtual_dir = virtual_dir
        self.lidar_sweeps = lidar_sweeps
        self.remove_ego_points = remove_ego_points
        self.command_fn = command_fn or db.command
        self.model = load_model()
        self.z_shift = float(self.model['ground_z_ego'] - GROUND_Z_EGO)
        self.renderer = VirtualCamRenderer(db, distort=distort)
        self._images = {}

    # ------------------------------------------------------------------ history / status
    def history_tokens(self, tok):
        """[tok(-num_history+1), ..., tok(0)], oldest first. Missing previous keyframes (scene
        start) are filled with the oldest available one."""
        toks = [tok]
        while len(toks) < self.num_history:
            p = self.db.prev_keyframe(toks[0])
            toks.insert(0, p if p is not None else toks[0])
        return toks

    def ego_statuses(self, tok, toks=None):
        toks = toks or self.history_tokens(tok)
        cmd = self.command_fn(tok)
        T0i = inv_T(self.db.ego2global(tok))
        out = []
        for h in toks:
            Tr = T0i @ self.db.ego2global(h)
            yaw = float(np.arctan2(Tr[1, 0], Tr[0, 0]))
            s = self.db.ego_state(h)
            out.append({
                'ego_pose': np.array([Tr[0, 3], Tr[1, 3], yaw], dtype=np.float32),
                'ego_velocity': np.array([s['vx'], s['vy']], dtype=np.float32),
                'ego_acceleration': np.array([s['ax'], s['ay']], dtype=np.float32),
                'driving_command': command_onehot(cmd),
            })
        return out, cmd

    # ------------------------------------------------------------------ lidar
    def _sweep_points_ego(self, info, T_cur_from_ego_sweep=None):
        pts = self.db.load_lidar_points(info['path'])
        T = info['sensor2ego'] if T_cur_from_ego_sweep is None else T_cur_from_ego_sweep @ info['sensor2ego']
        xyz = pts[:, :3].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        return xyz, pts[:, 3], pts[:, 4]

    def lidar_pc(self, tok):
        """(6, N) float32: x, y, z, intensity, ring, lidar_id in the NAVSIM ego frame."""
        info = self.db.lidar(tok)
        xyz, inten, ring = self._sweep_points_ego(info)
        parts = [(xyz, inten, ring)]
        if self.lidar_sweeps > 1:
            e2g_cur_inv = inv_T(self.db.ego2global(tok))
            for sw in self.db.sweeps(tok, 'LIDAR_TOP', self.lidar_sweeps - 1):
                if sw['ego2global'] is None:
                    continue
                parts.append(self._sweep_points_ego(sw, e2g_cur_inv @ sw['ego2global']))
        xyz = np.concatenate([p[0] for p in parts])
        inten = np.concatenate([p[1] for p in parts])
        ring = np.concatenate([p[2] for p in parts])
        if self.remove_ego_points:
            m = (np.abs(xyz[:, 0] - EGO_CENTER_X) < EGO_LENGTH / 2 + 0.25) & (np.abs(xyz[:, 1]) < EGO_WIDTH / 2 + 0.25)
            xyz, inten, ring = xyz[~m], inten[~m], ring[~m]
        xyz[:, 2] += self.z_shift
        pc = np.stack([xyz[:, 0], xyz[:, 1], xyz[:, 2], inten, ring, np.zeros(len(ring))], 0)
        return pc.astype(np.float32)

    # ------------------------------------------------------------------ cameras
    def virtual_path(self, tok, cam):
        if not self.virtual_dir:
            return None
        sub = 'distorted' if self.distort else 'pinhole'
        return os.path.join(self.virtual_dir, sub, self.db.scene_name(tok), cam, tok + '.jpg')

    def camera_image(self, tok, cam):
        """RGB uint8 1920x1080 virtual image of NAVSIM camera `cam` ('CAM_F0', ...)."""
        p = self.virtual_path(tok, cam)
        if p and os.path.exists(p):
            from PIL import Image
            return np.asarray(Image.open(p).convert('RGB'))
        return self.renderer.render(tok, cam)

    def camera_params(self, tok, cam):
        R, t, K, D = self.renderer.camera_params(tok, cam)
        t = t.copy()
        t[2] += self.z_shift
        return R, t, K, D

    # ------------------------------------------------------------------ AgentInput
    def agent_input(self, tok, sensor_config):
        from navsim.common import dataclasses as nd
        toks = self.history_tokens(tok)
        statuses, cmd = self.ego_statuses(tok, toks)
        cam_fields = {f.name for f in dataclasses.fields(nd.Camera)}
        ego_statuses, cameras, lidars = [], [], []
        for i, h in enumerate(toks):
            st = statuses[i]
            ego_statuses.append(nd.EgoStatus(ego_pose=st['ego_pose'], ego_velocity=st['ego_velocity'],
                                             ego_acceleration=st['ego_acceleration'],
                                             driving_command=st['driving_command']))
            names = sensor_config.get_sensors_at_iteration(i)
            cams = {}
            for attr in CAM_ATTRS:
                if attr in names:
                    cam = attr.upper()
                    R, t, K, D = self.camera_params(h, cam)
                    kw = dict(image=self.camera_image(h, cam), sensor2lidar_rotation=R,
                              sensor2lidar_translation=t, intrinsics=K, distortion=D)
                    if 'camera_path' in cam_fields:
                        kw['camera_path'] = self.virtual_path(h, cam)
                    cams[attr] = nd.Camera(**{k: v for k, v in kw.items() if k in cam_fields})
                else:
                    cams[attr] = nd.Camera()
            cameras.append(nd.Cameras(**cams))
            lidars.append(nd.Lidar(lidar_pc=self.lidar_pc(h)) if 'lidar_pc' in names else nd.Lidar())
        return nd.AgentInput(ego_statuses=ego_statuses, cameras=cameras, lidars=lidars), cmd
