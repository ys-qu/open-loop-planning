#!/usr/bin/env python
"""Impromptu-VLA (Qwen2.5-VL fine-tuned on nuScenes planning) zero-shot on the parking dataset.

Replicates the nuScenes planning query of the released `nuscenes_test.json`
(ahydchh/Impromptu-VLA@84f0f85) exactly:
  * image: virtual nuScenes CAM_FRONT (1600x900), pre-resized to <= 262144 px (LLaMA-Factory
    `image_max_pixels`, as in train/inference_scripts/sglang_infer.py) -> 682x384 -> 672x392;
  * LLaMA-Factory `qwen2_vl` template (default system prompt), <image> placeholder inline;
  * history: up to 3.0 s at 0.5 s of [x, y] of the (virtual) nuScenes LIDAR_TOP origin in the
    current LiDAR frame rotated to x-forward / y-left, plus CAN-like status per step
    (Acceleration X/Y, Velocity, Steering angle);
  * greedy decoding (sglang temperature 0.01 / top_k 1, repetition_penalty 1.0).
Output: 6 waypoints (0.5..3.0 s) of the LiDAR origin -> converted to the ego origin.
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vlm_common'))
import numpy as np  # noqa: E402

from olp_inputs import (NUSC_LIDAR_OFFSET, NUSC_VIRTUAL_INFOS, const_vel_fallback,  # noqa: E402
                       nusc_steering_from_state, parking_vocab, past_states, point_positions)
from olpcommon.geometry import ref_traj_to_ego  # noqa: E402
from runner import ParkingDataset, PredWriter, base_argparser, batched, eta, iterate  # noqa: E402

PREFIX = ("You are an autonomous driving agent. You have access to a front view camera image of a vehicle <image>. "
          "Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, given "
          "the vehicle's intent inferred from the images.")
HIST_HEAD = ("Provided are the previous ego vehicle status recorded over the last {T} seconds (at 0.5-second "
             "intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction "
             "while positive y means leftwards. The data is presented in the format [x, y]:.")
SYSTEM = "You are a helpful assistant."  # LLaMA-Factory qwen2_vl default_system
IMAGE_MAX_PIXELS = 262144
PT_RE = re.compile(r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]")
N_FUT, DT = 6, 0.5


def _num(x):
    """Formatting used by the Impromptu generator: python float of np.round(x, 2), -0.0 -> 0.0."""
    x = float(np.round(x, 2))
    return 0.0 if x == 0 else x


def build_user_content(entries):
    """entries: list (oldest first) of dicts x, y, ax, ay, v, steer; last one is t-0.0."""
    T = round(0.5 * len(entries) - 0.5, 1)
    s = PREFIX + HIST_HEAD.format(T=T)
    parts = []
    for i, e in enumerate(entries):
        parts.append("(t-%ss) [%s, %s], Acceleration: X %s, Y %s m/s^2, Velocity: %s m/s, Steering angle: %s "
                     "(positive: left turn, negative: right turn)" % (
                         round(T - i * 0.5, 1), _num(e['x']), _num(e['y']), round(float(e['ax']), 2),
                         round(float(e['ay']), 2), round(float(e['v']), 2), round(float(e['steer']), 2)))
    return s + ", ".join(parts) + "\n"


def history_entries(db, tok, hist_s=3.0):
    offs = [-0.5 * k for k in range(int(round(hist_s / 0.5)), -1, -1)]
    sts = [s for s in past_states(db, tok, offs) if s['valid']]
    p = point_positions(sts, NUSC_LIDAR_OFFSET) - np.asarray(NUSC_LIDAR_OFFSET)[None]
    out = []
    for s, xy in zip(sts, p):
        out.append(dict(x=xy[0], y=xy[1], ax=s['ax'], ay=s['ay'], v=s['vx'], steer=nusc_steering_from_state(s)))
    return out, sts[-1]


def chat_text(content):
    content = content.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    return ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
            % (SYSTEM, content))


def parse_waypoints(text):
    t = text
    if '<PLANNING>' in t:
        t = t[t.find('<PLANNING>') + len('<PLANNING>'):]
    if '</PLANNING>' in t:
        t = t[:t.find('</PLANNING>')]
    if '[x, y]:' in t:
        t = t[t.find('[x, y]:') + len('[x, y]:'):]
    return [[float(a), float(b)] for a, b in PT_RE.findall(t)]


def selftest():
    """Rebuild the user prompts of the released nuScenes test set from their own numbers."""
    p = os.path.join(HERE, '..', '..', 'third_party', 'Impromptu-VLA', 'nuscenes_test.json')
    d = json.load(open(p))
    pat = re.compile(r"\(t-[\d.]+s\) \[([-\d.e]+), ([-\d.e]+)\], Acceleration: X ([-\d.e]+), Y ([-\d.e]+) m/s\^2, "
                     r"Velocity: ([-\d.e]+) m/s, Steering angle: ([-\d.e]+)")
    bad = 0
    for x in d:
        c = x['messages'][0]['content']
        ents = [dict(x=float(a), y=float(b), ax=float(c_), ay=float(d_), v=float(e), steer=float(f))
                for a, b, c_, d_, e, f in pat.findall(c)]
        if build_user_content(ents) != c:
            bad += 1
            if bad < 3:
                print('MISMATCH\n', repr(c), '\n', repr(build_user_content(ents)))
    print('selftest: %d / %d prompts reproduced exactly' % (len(d) - bad, len(d)))
    a = d[4]['messages'][1]['content']
    print('parse:', parse_waypoints(a))


def main():
    ap = base_argparser(__doc__, 'impromptu_7b_ad_ft')
    ap.add_argument('--ckpt', default=os.path.join(HERE, '..', '..', 'ckpts', 'impromptu_vla', '7B_AD_finetune'))
    ap.add_argument('--dtype', default='fp16-vit32', choices=['fp16', 'fp16-vit32', 'fp32'])
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    import torch
    from qwen_vl import generate, load_qwen25vl, open_rgb, resize_max_pixels
    db = ParkingDataset(args.dataroot, args.scenes, load_agents=False)
    infos = {x['token']: x for x in json.load(open(NUSC_VIRTUAL_INFOS))['infos']}
    toks = iterate(args, db)
    ckpt = os.path.abspath(args.ckpt)
    info = {
        'checkpoint': ckpt, 'hf_repo': 'aaaaaap/ImpromptuVLAModel/' + os.path.basename(ckpt.rstrip('/')),
        'dtype': args.dtype, 'image': 'virtual nuScenes CAM_FRONT 1600x900 -> 682x384 (<=262144 px) -> 672x392',
        'prompt': 'nuscenes_test.json format (x-y mode, 3.0 s history with CAN-like status)',
        'steering': 'proxy: 15*atan(2.588*yaw_rate/speed) (no steering signal in the dataset)',
        'decoding': 'greedy, repetition_penalty=1.0', 'horizon_s': N_FUT * DT, 'K': 1,
        'output_point': 'virtual nuScenes LIDAR_TOP origin (0.94 m ahead of rear axle) -> ego origin',
        'privileged_inputs': 'none (the Impromptu nuScenes prompt has no navigation command)',
        'fallback': 'const_vel when < 6 waypoints parsed', 'every': args.every,
    }
    W = PredWriter(args, args.variant, info)
    todo = [t for t in toks if not W.done(t)]
    print('frames: %d requested, %d to do' % (len(toks), len(todo)))
    if not todo:
        W.save()
        return
    model, proc = load_qwen25vl(ckpt, args.dtype)
    print('model loaded; GPU mem %.1f GB' % (torch.cuda.memory_allocated() / 1e9))
    t_fut = np.round(np.arange(1, N_FUT + 1) * DT, 3)
    gen = dict(do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for bi, batch in enumerate(batched(todo, args.batch_size)):
        batch = [t for t in batch if not W.done(t)]  # another worker may have done them
        if not batch:
            continue
        texts, images, metas = [], [], []
        for t in batch:
            ents, st0 = history_entries(db, t)
            content = build_user_content(ents)
            p = infos[t]['cams']['CAM_FRONT']['data_path']
            im = resize_max_pixels(open_rgb(p), IMAGE_MAX_PIXELS)
            texts.append(chat_text(content))
            images.append([im])
            metas.append((content, p))
        outs, inputs = generate(model, proc, texts, images, args.max_new_tokens, gen)
        if bi == 0:
            print('image grid thw', inputs['image_grid_thw'].tolist()[:1], 'prompt tokens', inputs['input_ids'].shape)
            print('PROMPT:', metas[0][0])
            print('OUTPUT:', outs[0])
        for t, o, (content, p) in zip(batch, outs, metas):
            pts = parse_waypoints(o)
            extra = {'text': o, 'prompt': content, 'image': p, 'parking_vocab': parking_vocab(o)}
            if len(pts) >= N_FUT:
                xy_l = np.array(pts[:N_FUT])
                xy = ref_traj_to_ego(xy_l, NUSC_LIDAR_OFFSET)
                extra.update(parse_fail=False, lidar_xy=xy_l.tolist())
            else:
                xy = const_vel_fallback(db, t, t_fut)
                extra.update(parse_fail=True, fallback='const_vel', n_points=len(pts))
            W.add(t, t_fut, xy[None], extra=extra, raw={'text': o})
        if bi % 10 == 0:
            print('[%s] %s  peak mem %.1f GB' % (args.variant, eta(min((bi + 1) * args.batch_size, len(todo)), len(todo), t0),
                                                  torch.cuda.max_memory_allocated() / 1e9), flush=True)
    W.info['runtime_s_per_frame'] = (time.time() - t0) / len(todo)
    W.info['peak_gpu_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
    W.save()
    print('stats', W.stats())


if __name__ == '__main__':
    main()
