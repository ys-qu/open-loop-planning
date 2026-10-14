#!/usr/bin/env python
"""Positive control: run an agent of run_agent.py on REAL NAVSIM data (local mini split,
loaded with NAVSIM's own SceneLoader) and report open-loop L2 errors w.r.t. the human
trajectory. If an agent is wired up correctly (checkpoint, feature builders, output frame) it
must do well here; results on the parking dataset can then be read as pure domain shift.

  CUDA_VISIBLE_DEVICES=1 flock locks/gpu1.lock python methods/navsim/navsim_mini_check.py --agent transfuser
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_OLP_DERIVED = os.environ.get('OLP_DERIVED', os.path.join(os.path.join(HERE, '..', '..'), 'derived'))
sys.path.insert(0, HERE)
import run_agent  # noqa: E402

NAVSIM_ROOT = os.environ.get('NAVSIM_MINI_ROOT', '')  # local NAVSIM mini split (optional positive check)


def main():
    run_agent.load_external_agents()
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', required=True, choices=sorted(run_agent.AGENTS))
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--n', type=int, default=120, help='number of NAVSIM scenes')
    ap.add_argument('--split', default='mini')
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    spec = run_agent.AGENTS[args.agent]
    sys.path.insert(0, spec['code'])
    os.environ.setdefault('NUPLAN_MAPS_ROOT', os.path.join(NAVSIM_ROOT, 'maps'))
    os.environ.setdefault('OPENSCENE_DATA_ROOT', NAVSIM_ROOT)
    import torch
    from pathlib import Path
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from pyquaternion import Quaternion
    torch.manual_seed(0)

    agent = spec['build'](args.ckpt or spec['ckpt'])
    agent.eval()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.to(dev)
    kw = dict(data_path=Path(NAVSIM_ROOT) / 'navsim_logs' / args.split,
              scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True),
              sensor_config=agent.get_sensor_config())
    try:
        loader = SceneLoader(original_sensor_path=Path(NAVSIM_ROOT) / 'sensor_blobs' / args.split, **kw)
    except TypeError:  # navsim v1.x signature
        loader = SceneLoader(sensor_blobs_path=Path(NAVSIM_ROOT) / 'sensor_blobs' / args.split, **kw)
    toks = loader.tokens
    toks = [toks[i] for i in np.linspace(0, len(toks) - 1, min(args.n, len(toks))).astype(int)]
    builders = agent.get_feature_builders()
    if spec.get('batch_size'):
        args.batch_size = spec['batch_size']
    errs, n_back, cmds, min_ade, dx4 = [], 0, [], [], []
    t0 = time.time()
    for b0 in range(0, len(toks), args.batch_size):
        batch, gts = [], []
        for tok in toks[b0:b0 + args.batch_size]:
            ai = loader.get_agent_input_from_token(tok)
            ai.src_token = tok
            f = {}
            for b in builders:
                f.update(b.compute_features(ai))
            batch.append(f)
            frames = loader.scene_frames_dicts[tok]

            def se2(fr):
                q = Quaternion(*fr['ego2global_rotation'])
                return np.array(fr['ego2global_translation'][:2]), q.yaw_pitch_roll[0]
            p0, h0 = se2(frames[3])
            c, s = np.cos(h0), np.sin(h0)
            R = np.array([[c, -s], [s, c]])
            gts.append(np.array([R.T @ (se2(fr)[0] - p0) for fr in frames[4:12]]))
            cmds.append(int(np.argmax(frames[3]['driving_command'])))
        feats = run_agent.to_device(run_agent.collate([(f, None) for f in batch])[0], dev)
        with torch.no_grad():
            out = agent.forward(feats)
        for i, gt in enumerate(gts):
            trajs, _, _ = spec['post'](out, i)
            pr = trajs[0]
            errs.append(np.linalg.norm(pr - gt, axis=1))
            min_ade.append(np.linalg.norm(trajs - gt[None], axis=2).mean(1).min())
            dx4.append(pr[-1, 0] - gt[-1, 0])
            n_back += int(pr[3, 0] < -0.3)
    e = np.array(errs)
    print('[navsim_mini_check] agent=%s scenes=%d (%.0fs)  L2@1s %.2f  L2@2s %.2f  L2@3s %.2f  L2@4s %.2f  '
          'ADE(0-4s) %.2f  backward plans %d' % (args.agent, len(e), time.time() - t0, e[:, 1].mean(), e[:, 3].mean(),
                                                  e[:, 5].mean(), e[:, 7].mean(), e.mean(), n_back))
    print('  K=%d minADE %.2f   mean longitudinal error at 4 s (pred - GT) %+.2f m' % (
        len(trajs), np.mean(min_ade), np.mean(dx4)))
    print('commands used (0=L,1=S,2=R,3=U):', np.bincount(cmds, minlength=4).tolist())


if __name__ == '__main__':
    main()
