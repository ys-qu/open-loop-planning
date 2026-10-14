"""Small numpy-only geometry helpers (python>=3.7, no other deps)."""
import numpy as np

# ego_pose.json stores a body frame whose +x points to the REAR of the car.
# Right-multiplying its rotation by this matrix gives the x-forward / y-left
# ego frame used by calibrated_sensor.json and by the agent boxes.
FLIP_Z180 = np.diag([-1.0, -1.0, 1.0])


def quat_to_rot(q):
    """[w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    """3x3 rotation matrix -> [w, x, y, z] (w >= 0)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def slerp(q0, q1, a):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot2d(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def transform_points(T, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def box_corners_2d(cx, cy, length, width, yaw):
    """4x2 corners of an oriented rectangle."""
    dx, dy = length / 2.0, width / 2.0
    c = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    return c @ rot2d(yaw).T + np.array([cx, cy])


def _axes(poly):
    e = np.roll(poly, -1, axis=0) - poly
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def polygons_intersect(a, b):
    """Separating-axis test for two convex polygons (Nx2 arrays)."""
    for ax in np.vstack([_axes(a), _axes(b)]):
        pa, pb = a @ ax, b @ ax
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


def headings_from_xy(xy, min_step=0.05, min_turn_radius=4.5):
    """Heading along a planar trajectory that starts at the origin with yaw 0.

    Segments that point backwards (> 90 deg from the current heading) are treated as
    reversing, so the heading is kept instead of flipped. The heading change per segment is
    limited to ds / min_turn_radius (a car cannot rotate on the spot), which keeps
    near-stationary, noisy trajectories from producing spurious +-90 deg headings.
    xy: [T, 2] -> [T]."""
    pts = np.vstack([[0.0, 0.0], np.asarray(xy, dtype=np.float64)])
    yaw, out = 0.0, []
    for k in range(1, len(pts)):
        d = pts[k] - pts[k - 1]
        ds = np.linalg.norm(d)
        if ds > min_step:
            a = np.arctan2(d[1], d[0])
            if abs(wrap_angle(a - yaw)) > np.pi / 2:
                a = wrap_angle(a + np.pi)
            max_turn = ds / min_turn_radius
            yaw = float(wrap_angle(yaw + np.clip(wrap_angle(a - yaw), -max_turn, max_turn)))
        out.append(yaw)
    return np.array(out)


def ref_traj_to_ego(xy_ref, ref_offset, yaw=None):
    """Convert a trajectory of a vehicle-fixed reference point to the ego origin.

    xy_ref: [T, 2] future positions of a point that sits at `ref_offset` = (dx, dy) in the
    ego frame (e.g. a LiDAR 0.94 m ahead of the rear axle), expressed in the CURRENT ego
    frame *relative to the point's current position*, i.e. xy_ref[t] = p_ref(t) - p_ref(0).
    yaw: optional [T] headings; estimated from the trajectory if None.
    Returns [T, 2] positions of the ego origin in the current ego frame."""
    xy_ref = np.asarray(xy_ref, dtype=np.float64)
    off = np.asarray(ref_offset, dtype=np.float64)
    if yaw is None:
        yaw = headings_from_xy(xy_ref)
    p_abs = xy_ref + off  # reference point positions in the current ego frame
    out = np.zeros_like(p_abs)
    for k in range(len(p_abs)):
        out[k] = p_abs[k] - rot2d(yaw[k]) @ off
    return out
