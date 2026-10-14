"""Virtual nuPlan/NAVSIM cameras rendered from the pinhole cameras of the parking dataset.

NAVSIM images (OpenScene / nuPlan sensor blobs) are RAW 1920x1080 images with strong
barrel distortion (K: f=1545, c=(960, 560); OpenCV D=[-0.356, 0.173, -0.002, 0.0005,
-0.052]). We checked this on the local NAVSIM mini split: LiDAR depth edges projected with
K+D land on image edges clearly better than with the pinhole K alone (better in 90% of 138
images). The NAVSIM models were trained on these distorted images and use fixed crops
(TransFuser: l0/r0 [28:-28, 416:-416], f0 [28:-28]), so we synthesise exactly that camera:

  for every output pixel (u, v) of virtual camera CAM_X:
     ray_cam  = undistort(K_nuplan, D_nuplan, (u, v))            (OpenCV model, iterative)
     ray_ego  = R_nuplan(CAM_X) @ ray_cam                         (mean NAVSIM mounting rotation)
     source   = the main source camera (NAVSIM_TO_SRC[CAM_X][0]); pixels it does not see
                (outside the image or the circular vignette, r > 1350 px) are taken from the
                next camera in NAVSIM_TO_SRC[CAM_X] (only needed for B0/L2/R2 corners)
     (u', v') = K_src @ R_src^T @ ray_ego                     (source images are pinhole)
  image = bilinear remap.

Only the ROTATION of the nuPlan camera is reproduced (the optical centre stays at the source
camera: rays are treated as directions, i.e. a pure-rotation homography + lens model).
The nuPlan front cameras sit ~1.85 m above ground, the source ones 2.01-2.05 m (see README).
With distort=False the output is a pinhole camera with the nuPlan K (no lens distortion).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(HERE, 'nuplan_cam_model.json')

# NAVSIM camera -> preferred source camera(s) (by mounting yaw).
# Source cameras: CAM_5 front (2.9 deg), CAM_1 front-left (50.8), CAM_2 front-right (-48.4),
#        CAM_4 back-left (109.2), CAM_3 back-right (-107.0), CAM_6 back (-179.3)
NAVSIM_TO_SRC = {
    'CAM_F0': ['CAM_5', 'CAM_1', 'CAM_2'],
    'CAM_L0': ['CAM_1', 'CAM_5', 'CAM_4'],
    'CAM_R0': ['CAM_2', 'CAM_5', 'CAM_3'],
    'CAM_L1': ['CAM_4', 'CAM_1'],
    'CAM_R1': ['CAM_3', 'CAM_2'],
    'CAM_L2': ['CAM_4', 'CAM_6'],
    'CAM_R2': ['CAM_3', 'CAM_6'],
    'CAM_B0': ['CAM_6', 'CAM_4', 'CAM_3'],
}
SRC_VALID_RADIUS = 1350.0  # px from the principal point; the vignette starts at ~1400 px


def load_model(path=MODEL_JSON):
    with open(path) as f:
        m = json.load(f)
    m['K'] = np.array(m['K'], dtype=np.float64)
    m['D'] = np.array(m['D'], dtype=np.float64)
    for c in m['cams'].values():
        c['R'] = np.array(c['sensor2ego_rotation'], dtype=np.float64)
        c['t'] = np.array(c['sensor2ego_translation'], dtype=np.float64)
    return m


def _undistort_grid(K, D, W, H, distort):
    import cv2
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel()], 1)
    if distort:
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
        xy = cv2.undistortPointsIter(pts.reshape(-1, 1, 2), K, D, None, None, crit).reshape(-1, 2)
        # sanity: re-distort and check round trip
        rt, _ = cv2.projectPoints(np.concatenate([xy, np.ones((len(xy), 1))], 1).reshape(-1, 1, 3),
                                  np.zeros(3), np.zeros(3), K, D)
        err = np.abs(rt.reshape(-1, 2) - pts).max()
        assert err < 0.05, 'undistortion round-trip error %.3f px' % err
    else:
        xy = (pts - K[:2, 2]) / np.diag(K)[:2]
    return np.concatenate([xy, np.ones((len(xy), 1))], 1)  # [H*W, 3] rays in the camera frame


class VirtualCamRenderer(object):
    """Renders virtual NAVSIM cameras from the dataset keyframe cameras.

    db: olpcommon.ParkingDataset; distort: reproduce the nuPlan lens distortion (default True)."""

    def __init__(self, db, distort=True, model_path=MODEL_JSON, valid_radius=SRC_VALID_RADIUS):
        self.db = db
        self.m = load_model(model_path)
        self.distort = distort
        self.W, self.H = int(self.m['width']), int(self.m['height'])
        self.valid_radius = valid_radius
        self._rays = None
        self._maps = {}

    def rays(self):
        if self._rays is None:
            self._rays = _undistort_grid(self.m['K'], self.m['D'], self.W, self.H, self.distort)
        return self._rays

    def camera_params(self, tok, cam):
        """(sensor2ego rotation, sensor2ego translation, K, D) of virtual camera `cam`, in the
        dataset ego frame. Rotation = mean NAVSIM mounting rotation; translation = position
        of the main source camera (the true optical centre of the synthesised view)."""
        D = self.m['D'] if self.distort else np.zeros(5)
        return (self.m['cams'][cam]['R'].copy(), self.source_translation(tok, cam).copy(),
                self.m['K'].copy(), np.asarray(D, dtype=np.float64).copy())

    def maps(self, tok, cam):
        """Per source camera: (channel, map_x, map_y, mask), cached by calibration tokens."""
        srcs = [self.db.camera(tok, s) for s in NAVSIM_TO_SRC[cam]]
        key = (cam,) + tuple((s['channel'], np.round(s['sensor2ego'], 6).tobytes(), np.round(s['K'], 4).tobytes())
                             for s in srcs)
        if key in self._maps:
            return self._maps[key]
        rays_ego = self.rays() @ self.m['cams'][cam]['R'].T  # [N, 3]
        rays_ego /= np.linalg.norm(rays_ego, axis=1, keepdims=True)
        best = np.full(len(rays_ego), -1, dtype=np.int64)
        uvs = []
        for i, s in enumerate(srcs):
            R, K = s['sensor2ego'][:3, :3], s['K']
            d = rays_ego @ R  # rays in the source camera frame (R^T d)
            z = d[:, 2]
            with np.errstate(divide='ignore', invalid='ignore'):
                uv = (d[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
            ok = (z > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] <= s['width'] - 1) & \
                 (uv[:, 1] >= 0) & (uv[:, 1] <= s['height'] - 1) & \
                 (np.hypot(uv[:, 0] - K[0, 2], uv[:, 1] - K[1, 2]) < self.valid_radius)
            take = ok & (best < 0)  # priority = order of NAVSIM_TO_SRC[cam] (primary camera first)
            best[take] = i
            uvs.append(uv)
        out = []
        for i, s in enumerate(srcs):
            m = best == i
            if not m.any():
                continue
            mx = np.where(m, uvs[i][:, 0], -1).reshape(self.H, self.W).astype(np.float32)
            my = np.where(m, uvs[i][:, 1], -1).reshape(self.H, self.W).astype(np.float32)
            out.append((s['channel'], mx, my, m.reshape(self.H, self.W)))
        self._maps[key] = (out, float((best >= 0).mean()))
        return self._maps[key]

    def render(self, tok, cam, images=None):
        """RGB uint8 [H, W, 3] virtual view of NAVSIM camera `cam` at keyframe `tok`.

        images: optional dict channel -> RGB array (to reuse decoded source images)."""
        import cv2
        from PIL import Image
        parts, _ = self.maps(tok, cam)
        out = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        for ch, mx, my, m in parts:
            if images is not None and ch in images:
                src = images[ch]
            else:
                src = np.asarray(Image.open(self.db.camera(tok, ch)['path']).convert('RGB'))
                if images is not None:
                    images[ch] = src
            warped = cv2.remap(src, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            out[m] = warped[m]
        return out

    def source_translation(self, tok, cam):
        """Position (ego frame) of the main source camera of virtual camera `cam`."""
        return self.db.camera(tok, NAVSIM_TO_SRC[cam][0])['sensor2ego'][:3, 3]
