"""GTRS-Dense and Hydra-MDP (image-only, V2-99) agent specs for methods/navsim/run_agent.py.

Code: third_party/GTRS (NVlabs/GTRS @ 92a740d, NAVSIM v2 fork). Checkpoints: HF Zzxxxxxxxx/gtrs
(hydra_mdp_vov.ckpt; dd3d_det_final.pth is only the V2-99 init that the
model constructor insists on loading before the full checkpoint overwrites it).

Both are trajectory-SCORING planners: a fixed vocabulary of 16384 trajectories (traj_final/
16384.npy, 4 s @ 10 Hz, x/y/heading of the rear axle) is scored by heads that imitate the
PDM sub-scores + an imitation head, conditioned on the stitched l0|f0|r0 image (2048x512, same
crop as TransFuser) and the ego status (command, velocity, acceleration). Top-1 = argmax of the
repo's hard-coded inference score (HydraTrajHead.forward). We store the top-32 vocabulary
trajectories by that score (trajs[0] == the model's own 'trajectory'), sub-sampled to 2 Hz
(t = 0.5..4.0 s) to match the other NAVSIM agents (the 10 Hz plan is a piecewise-smooth curve,
so linear re-interpolation by eval_planning.py loses < 1 cm).
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(ROOT, 'third_party', 'GTRS')
CKPT_DIR = os.path.join(ROOT, 'ckpts', 'gtrs')
K_OUT = 32


def build(ckpt, vocab_dropout):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.gtrs_dense.gtrs_agent import GTRSAgent
    from navsim.agents.gtrs_dense.hydra_config import HydraConfig
    # = navsim/planning/script/config/common/agent/hydra_mdp_vov.yaml
    cfg = HydraConfig(
        vocab_path=os.path.join(CODE, 'traj_final', '16384.npy'), ckpt_path='unused', vocab_size=16384,
        lidar_seq_len=4, sigma=0.5, trajectory_imi_weight=1.0, progress_weight=2.0,
        vocab_dropout=vocab_dropout, normalize_vocab_pos=True, camera_width=2048, camera_height=512,
        img_vert_anchors=16, img_horz_anchors=64, backbone_type='vov',
        vov_ckpt=os.path.join(CKPT_DIR, 'dd3d_det_final.pth'), lr_mult_backbone=1.0,
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.1))
    agent = GTRSAgent(cfg, lr=1e-4, checkpoint_path=ckpt, pdm_gt_path=None)
    agent.initialize()  # strict=False; only the vocab buffer is dropped from the ckpt
    return agent


def scores_from_heads(r):
    """Same formula as HydraTrajHead.forward (GTRS @ 92a740d)."""
    return (0.03 * r['imi'].softmax(-1).log() +
            0.1 * r['traffic_light_compliance'].sigmoid().log() +
            0.1 * r['no_at_fault_collisions'].sigmoid().log() +
            0.9 * r['drivable_area_compliance'].sigmoid().log() +
            0.2 * r['driving_direction_compliance'].sigmoid().log() +
            6.0 * (7.0 * r['time_to_collision_within_bound'].sigmoid() +
                   7.0 * r['ego_progress'].sigmoid() +
                   3.0 * r['lane_keeping'].sigmoid()).log())


def post(out, i):
    import torch
    sc = scores_from_heads({k: out[k][i:i + 1] for k in ['imi', 'traffic_light_compliance', 'no_at_fault_collisions',
                                                         'drivable_area_compliance', 'driving_direction_compliance',
                                                         'time_to_collision_within_bound', 'ego_progress',
                                                         'lane_keeping']})[0]
    top_s, top_i = torch.topk(sc, K_OUT)
    assert int(top_i[0]) == int(out['selected_indices'][i])
    vocab = out['trajectory_vocab']  # [16384, 40, 3]
    tr = vocab[top_i].float().cpu().numpy()[:, 4::5]  # 10 Hz -> 2 Hz (0.5 .. 4.0 s)
    top = out['trajectory'][i].float().cpu().numpy()
    assert np.allclose(tr[0], top[4::5], atol=1e-5)
    return tr[:, :, :2], top_s.float().cpu().numpy(), {'heading': np.round(top[4::5, 2], 4).tolist(),
                                                       'vocab_idx_top': top_i[:8].cpu().tolist()}


SENSORS = ['cam_f0', 'cam_l0', 'cam_r0', 'cam_b0', 'cam_l2', 'cam_r2']  # builder also stitches l2|b0|r2 (unused)
AGENTS = {
    'hydra_mdp': dict(code=CODE, ckpt=os.path.join(CKPT_DIR, 'hydra_mdp_vov.ckpt'),
                      build=lambda ck: build(ck, vocab_dropout=False), post=post, sensors=SENSORS, sensor_hist=[2, 3]),
}
