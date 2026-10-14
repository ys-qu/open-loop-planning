#!/usr/bin/env python
"""Zero-shot inference of NAVSIM-trained planners on the dataset keyframes.

  CUDA_VISIBLE_DEVICES=0 $CONDA_BASE/envs/olp_navsim/bin/python methods/navsim/run_agent.py \
      --agent transfuser --variant navsim_transfuser

Agents (code base that is put on sys.path first, checkpoint):
  transfuser      third_party/navsim (v2.2)      ckpts/navsim_baselines/transfuser/transfuser_seed_0.ckpt
  ego_mlp         third_party/navsim (v2.2)      ckpts/navsim_baselines/ego_status_mlp/ego_status_mlp_seed_0.ckpt
  diffusiondrive  third_party/DiffusionDrive_navsim  (registered in methods/diffusiondrive/agents.py)
  hydra_mdp       third_party/GTRS               (registered in methods/gtrs/agents.py)

Writes outputs/<variant>/predictions.json (olpcommon format): t = 0.5..4.0 s (2 Hz, NAVSIM
TrajectorySampling(4 s, 0.5 s)), trajs = K x 8 x 2 rear-axle (x fwd, y left) positions in the
ego frame at the keyframe; extra = command, headings, (scores / modes for multi-modal agents).
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from olp_navsim import ROOT  # noqa: E402  (also puts the project root on sys.path)
from olpcommon import DEFAULT_DATAROOT, ParkingDataset, save_predictions  # noqa: E402
from olpcommon.predictions import make_entry  # noqa: E402

TP = os.path.join(ROOT, 'third_party')
CK = os.path.join(ROOT, 'ckpts')


# ---------------------------------------------------------------------------- agent registry
def _build_transfuser(ckpt, latent):
    from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
    from navsim.agents.transfuser.transfuser_config import TransfuserConfig
    agent = TransfuserAgent(TransfuserConfig(latent=latent), lr=1e-4, checkpoint_path=ckpt)
    agent.initialize()  # strict load of the Lightning state_dict ("agent." prefix stripped)
    return agent


def _build_ego_mlp(ckpt):
    from navsim.agents.ego_status_mlp_agent import EgoStatusMLPAgent
    agent = EgoStatusMLPAgent(hidden_layer_dim=512, lr=1e-4, checkpoint_path=ckpt)
    agent.initialize()
    return agent


def _single_traj(out, i):
    tr = out['trajectory'][i].float().cpu().numpy()  # [8, 3] x, y, heading
    return tr[None, :, :2], None, {'heading': np.round(tr[:, 2], 4).tolist()}


AGENTS = {
    'transfuser': dict(code=os.path.join(TP, 'navsim'),
                       ckpt=os.path.join(CK, 'navsim_baselines/transfuser/transfuser_seed_0.ckpt'),
                       build=lambda ck: _build_transfuser(ck, latent=False), post=_single_traj),
    'ego_mlp': dict(code=os.path.join(TP, 'navsim'),
                    ckpt=os.path.join(CK, 'navsim_baselines/ego_status_mlp/ego_status_mlp_seed_0.ckpt'),
                    build=_build_ego_mlp, post=_single_traj),
}


NAVSIM_METHOD_DIRS = ['diffusiondrive', 'gtrs']  # methods/<m>/agents.py add DiffusionDrive and Hydra-MDP


def load_external_agents():
    """methods/<m>/agents.py (m in NAVSIM_METHOD_DIRS) defines AGENTS = {name: spec} (same keys as above)."""
    mdir = os.path.join(ROOT, 'methods')
    for m in NAVSIM_METHOD_DIRS:
        p = os.path.join(mdir, m, 'agents.py')
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location('olp_agents_%s' % m, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            AGENTS.update(getattr(mod, 'AGENTS', {}))


# ---------------------------------------------------------------------------- data
def current_frame_sensors(names, idxs=(3,)):
    """SensorConfig loading only `names` (e.g. cam_f0, cam_l0, cam_r0, lidar_pc) at history indices idxs."""
    from navsim.common.dataclasses import SensorConfig
    fields = ['cam_f0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0', 'lidar_pc']
    return SensorConfig(**{f: (list(idxs) if f in names else False) for f in fields})


class KeyframeDataset(object):
    def __init__(self, inp, tokens, agent, spec=None):
        self.inp, self.tokens = inp, tokens
        # an agent spec may restrict the sensors to those its feature builders really read
        # (e.g. DiffusionDrive requests all 8 cameras but only uses l0/f0/r0 + LiDAR)
        if spec and spec.get('sensors'):
            self.sensor_config = current_frame_sensors(spec['sensors'], spec.get('sensor_hist', (3,)))
        else:
            self.sensor_config = agent.get_sensor_config()
        self.builders = agent.get_feature_builders()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        import torch
        tok = self.tokens[i]
        ai, cmd = self.inp.agent_input(tok, self.sensor_config)
        ai.src_token = tok  # sample token, available to feature builders
        feats = {}
        for b in self.builders:
            feats.update(b.compute_features(ai))
        st = ai.ego_statuses[-1]
        meta = {'token': tok, 'command': cmd,
                'velocity': [float(v) for v in st.ego_velocity],
                'acceleration': [float(v) for v in st.ego_acceleration]}
        return feats, meta


def _collate(vals):
    import torch
    v0 = vals[0]
    if torch.is_tensor(v0):
        return torch.stack(vals)
    if isinstance(v0, np.ndarray):
        return torch.from_numpy(np.stack(vals))
    if isinstance(v0, (list, tuple)):  # e.g. GTRS: list over history frames / ego-status entries
        return [_collate([v[j] for v in vals]) for j in range(len(v0))]
    return vals


def collate(batch):
    feats = {k: _collate([b[0][k] for b in batch]) for k in batch[0][0]}
    return feats, [b[1] for b in batch]


def to_device(x, dev):
    import torch
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return [to_device(v, dev) for v in x]
    if isinstance(x, dict):
        return {k: to_device(v, dev) for k, v in x.items()}
    return x


def main():
    load_external_agents()
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', required=True, choices=sorted(AGENTS))
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--variant', default=None, help='output name (default: agent name)')
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    ap.add_argument('--out-dir', default=os.environ.get('OLP_OUTPUTS', os.path.join(ROOT, 'outputs')))
    ap.add_argument('--no-distort', action='store_true', help='pinhole virtual cameras (ablation)')
    ap.add_argument('--lidar-sweeps', type=int, default=1)
    ap.add_argument('--keep-ego-points', action='store_true')
    ap.add_argument('--command', default='gt', choices=['gt', 'straight', 'unknown'],
                    help="gt = ParkingDataset.command() (privileged); straight/unknown = fixed command")
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0, help='debug: first N keyframes only')
    ap.add_argument('--fp16', action='store_true', help='autocast fp16 (default fp32)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dump-aux', default=None, help='npz with raw model outputs for --aux-tokens')
    ap.add_argument('--aux-tokens', nargs='*', default=[])
    args = ap.parse_args()

    spec = AGENTS[args.agent]
    sys.path.insert(0, spec['code'])
    import torch
    import navsim  # noqa: F401
    print('[run_agent] navsim package from', os.path.dirname(navsim.__file__))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    db = ParkingDataset(args.dataroot, args.scenes, load_agents=False)
    from olp_navsim import OLPNavsimInputs
    cmd_fn = None
    if args.command != 'gt':
        cmd_fn = (lambda tok, c=args.command.upper(): c)
    inp = OLPNavsimInputs(db, distort=not args.no_distort, lidar_sweeps=args.lidar_sweeps,
                         remove_ego_points=not args.keep_ego_points, command_fn=cmd_fn)
    ckpt = args.ckpt or spec['ckpt']
    agent = spec['build'](ckpt)
    agent.eval()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.to(dev)
    tokens = db.keyframes[:args.limit] if args.limit else db.keyframes
    ds = KeyframeDataset(inp, tokens, agent, spec)
    print('[run_agent] sensor config:', ds.sensor_config)
    if spec.get('batch_size') and spec['batch_size'] != args.batch_size:
        print('[run_agent] %s runs with batch size %d (set in its agent spec)' % (args.agent, spec['batch_size']))
        args.batch_size = spec['batch_size']
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, collate_fn=collate)
    preds, aux = {}, {}
    t_sampling = np.round(np.arange(1, 9) * 0.5, 3)
    t0, n_fw = time.time(), 0.0
    torch.cuda.reset_peak_memory_stats() if dev.type == 'cuda' else None
    for feats, metas in loader:
        feats = to_device(feats, dev)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        tf = time.time()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=args.fp16):
            out = agent.forward(feats)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        n_fw += time.time() - tf
        for i, m in enumerate(metas):
            trajs, scores, extra = spec['post'](out, i)
            T = trajs.shape[1]
            t = t_sampling[:T] if T == 8 else np.round(np.arange(1, T + 1) * (4.0 / T), 3)
            extra = dict(extra or {})
            extra.update({'command': m['command'], 'ego_velocity': np.round(m['velocity'], 3).tolist(),
                          'ego_acceleration': np.round(m['acceleration'], 3).tolist()})
            preds[m['token']] = make_entry(t, trajs, scores=scores, extra=extra)
            if m['token'] in args.aux_tokens:
                aux[m['token']] = {k: v[i].float().cpu().numpy() for k, v in out.items()
                                   if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == len(metas)}
        if len(preds) % (args.batch_size * 20) == 0:
            print('  %d/%d  %.1fs' % (len(preds), len(tokens), time.time() - t0))
    wall = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == 'cuda' else 0
    variant = args.variant or args.agent
    used = set(n for i in range(inp.num_history) for n in ds.sensor_config.get_sensors_at_iteration(i))
    cams_used = sorted(n for n in used if n.startswith('cam_'))
    lidar_used = 'lidar_pc' in used
    info = {
        'agent': args.agent, 'checkpoint': os.path.relpath(ckpt, ROOT), 'code': os.path.relpath(spec['code'], ROOT),
        'cameras': ('virtual nuPlan %s (%s), see methods/navsim/README.md' % (
            ','.join(cams_used), 'pinhole' if args.no_distort else 'distorted')) if cams_used else 'not used',
        'lidar': ('%d sweep(s), NAVSIM ego frame (z+%.2f m), ego points %s' % (
            args.lidar_sweeps, inp.z_shift, 'kept' if args.keep_ego_points else 'removed')) if lidar_used else 'not used',
        'command': args.command, 'horizon_s': 4.0, 'dt': 0.5, 'fp16': args.fp16, 'seed': args.seed,
        'runtime_s': round(wall, 1), 'model_forward_s_per_frame': round(n_fw / max(len(preds), 1), 4),
        'peak_gpu_mem_MiB': round(mem, 0), 'batch_size': args.batch_size,
    }
    print('[run_agent] info', json.dumps(info))
    save_predictions(os.path.join(args.out_dir, variant, 'predictions.json'), variant, preds, info=info)
    if args.dump_aux and aux:
        np.savez_compressed(args.dump_aux, **{'%s__%s' % (tok, k): v for tok, d in aux.items() for k, v in d.items()})
        print('aux outputs written to', args.dump_aux)


if __name__ == '__main__':
    main()
