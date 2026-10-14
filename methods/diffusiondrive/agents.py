"""DiffusionDrive (NAVSIM, CVPR'25) agent spec for methods/navsim/run_agent.py.

Code: third_party/DiffusionDrive_navsim (separate clone of hustvl/DiffusionDrive @ 9b52ed0, main
branch = NAVSIM v1.1 fork; the shared third_party/DiffusionDrive checkout is left alone).
Checkpoint: HF hustvl/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS (+ kmeans anchors from the
GitHub release). Inference = the repo's forward_test (truncated diffusion, 2 DDIM steps from 20
noised anchors, top-1 = argmax of the classification head); we only additionally return all 20
denoised trajectories and their scores (logits) so they can be stored as K=20 modes.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(ROOT, 'third_party', 'DiffusionDrive_navsim')
CKPT_DIR = os.path.join(ROOT, 'ckpts', 'diffusiondrive')


def forward_test_all_modes(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img):
    """Copy of navsim/agents/diffusiondrive/transfuser_model_v2.py TrajectoryHead.forward_test
    (DiffusionDrive @ 9b52ed0) that also returns all modes ('all_trajectories' B x 20 x 8 x 3,
    'all_scores' B x 20 logits). The top-1 'trajectory' is unchanged."""
    import torch
    from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position
    step_num = 2
    bs = ego_query.shape[0]
    device = ego_query.device
    self.diffusion_scheduler.set_timesteps(1000, device)
    step_ratio = 20 / step_num
    roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
    roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
    plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
    img = self.norm_odo(plan_anchor)
    noise = torch.randn(img.shape, device=device)
    trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
    img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
    ego_fut_mode = img.shape[1]
    for k in roll_timesteps[:]:
        x_boxes = torch.clamp(img, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(x_boxes)
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        timesteps = k
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(img.device)
        timesteps = timesteps.expand(img.shape[0])
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs, 1, -1)
        poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                                                           agents_query, ego_query, time_embed, status_encoding, global_img)
        poses_reg = poses_reg_list[-1]
        poses_cls = poses_cls_list[-1]
        x_start = poses_reg[..., :2]
        x_start = self.norm_odo(x_start)
        img = self.diffusion_scheduler.step(model_output=x_start, timestep=k, sample=img).prev_sample
    mode_idx = poses_cls.argmax(dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
    best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
    return {'trajectory': best_reg, 'all_trajectories': poses_reg, 'all_scores': poses_cls}


def build(ckpt):
    import types
    from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
    cfg = TransfuserConfig(plan_anchor_path=os.path.join(CKPT_DIR, 'kmeans_navsim_traj_20.npy'))
    agent = TransfuserAgent(cfg, lr=6e-4, checkpoint_path=ckpt)  # loads the ckpt (strict=False, prints mismatches)
    head = agent._transfuser_model._trajectory_head
    head.forward_test = types.MethodType(forward_test_all_modes, head)
    return agent


def post(out, i):
    """K=20 modes sorted by score (trajs[0] == the model's own top-1 'trajectory')."""
    trajs = out['all_trajectories'][i].float().cpu().numpy()  # [20, 8, 3]
    sc = out['all_scores'][i].float().cpu().numpy()  # [20]
    order = np.argsort(-sc)
    top = out['trajectory'][i].float().cpu().numpy()
    assert np.allclose(trajs[order[0]], top, atol=1e-5)
    return trajs[order, :, :2], sc[order], {'heading': np.round(top[:, 2], 4).tolist(),
                                            'anchor_idx_sorted': order.tolist()}


AGENTS = {
    'diffusiondrive': dict(code=CODE, ckpt=os.path.join(CKPT_DIR, 'diffusiondrive_navsim_88p1_PDMS'),
                           build=build, post=post, sensors=['cam_f0', 'cam_l0', 'cam_r0', 'lidar_pc']),
}
