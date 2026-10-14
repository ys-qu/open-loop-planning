#!/usr/bin/env python
"""OpenEMMA-style training-free planning with Qwen2.5-VL on the dataset keyframes.

Pipeline per keyframe (OpenEMMA main.py, method "openemma", Qwen code path):
  1. three free-text VLM calls on the current front image: scene description, critical
     objects, driving intent (OpenEMMA prompts verbatim, max_new_tokens=128);
  2. a motion call: OpenEMMA prompt with the three texts + the past 5 s of
     [speed, curvature*100] at 0.5 s steps -> the model writes 10 future [speed, curvature]
     pairs (0.5 s steps, 5 s);
  3. the pairs are integrated (trapezoid, dt=0.5 s, from the current causal ego state) into
     ego-frame waypoints.
--mode direct skips step 1 (OpenEMMA's non-"openemma" baseline prompt).

See methods/openemma_qwen/README.md for the deviations from OpenEMMA and the exact prompts.
"""
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'vlm_common'))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from olp_inputs import (NUSC_VIRTUAL_INFOS, const_vel_fallback, curvature_from_state,  # noqa: E402
                       integrate_speed_curvature, parking_vocab, past_states)
from runner import ParkingDataset, PredWriter, base_argparser, batched, eta, iterate  # noqa: E402

# ----------------------------------------------------------------------------- prompts
# OpenEMMA (taco-group/OpenEMMA@8403ea6, main.py) prompts, non-llava versions, verbatim.
P_SCENE = ("You are a autonomous driving labeller. You have access to these front-view camera images of a car "
           "taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the "
           "driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.")
P_OBJECTS = ("You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle "
             "taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other "
             "road users should you pay attention to in the driving scene? List two or three of them, specifying "
             "its location within the image of the driving scene and provide a short description of the that road "
             "user on what it is doing, and why it is important to you.")
P_INTENT = ("You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle "
            "taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the "
            "lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego "
            "car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the "
            "current speed or slow down or speed up?")
SYS_MOTION = ("You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a "
              "sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature "
              "is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A "
              "positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The "
              "larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving "
              "straight. As a driver on the road, you should follow any common sense traffic rules. You should try to "
              "stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You "
              "should observe lane markings and follow them.  Your task is to do your best to predict future speeds "
              "and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the "
              "image. Make a best guess if the problem is too difficult for you. If you cannot provide a response "
              "people will get injured.\n")
# Our addition (the only change to the system message): units + reversing.
SYS_ADDITION = "The speed v is in m/s; a negative v means the vehicle is moving backwards (reversing).\n"

MOTION_OPENEMMA = """These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval.
        The scene is described as follows: {scene}.
        The identified critical objects are {objects}.
        The car's intent is {intent}.
        The 5 second historical velocities and curvatures of the ego car are {hist}.
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
MOTION_DIRECT = """These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval.
        The 5 second historical velocities and curvatures of the ego car are {hist}.
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""

PAIR_RE = re.compile(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]")
OBS_OFFSETS = [-0.5 * k for k in range(9, -1, -1)]  # 10 past samples at 2 Hz: t-4.5 ... t0
N_FUT, DT = 10, 0.5
MIN_PAIRS = 6  # >= 3 s of output required, shorter outputs are padded by holding the last pair


def history_string(db, tok):
    sts = past_states(db, tok, OBS_OFFSETS)
    items = []
    for s in sts:
        v = s['speed'] if s['valid'] else 0.0
        k = curvature_from_state(s) * 100.0 if s['valid'] else 0.0
        items.append("[%.1f,%.1f]" % (v, k))  # OpenEMMA format: f"[{x[0]:.1f},{x[1]:.1f}]"
    return ", ".join(items), sts[-1]


def parse_pairs(text):
    t = text.replace("Future speeds and curvatures:", "").strip()
    return [[float(a), float(b)] for a, b in PAIR_RE.findall(t)]


def image_for(tok, view, db, infos, max_pixels):
    from qwen_vl import open_rgb, resize_max_pixels
    if view == 'nusc':
        p = infos[tok]['cams']['CAM_FRONT']['data_path']
    else:
        p = db.camera(tok, 'CAM_FRONT')['path']
    im = open_rgb(p)
    if max_pixels:
        im = resize_max_pixels(im, max_pixels)
    return im, p


def main():
    ap = base_argparser(__doc__, 'openemma_qwen25vl7b')
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--dtype', default='fp16-vit32', choices=['fp16', 'fp16-vit32', 'fp32'])
    ap.add_argument('--mode', default='openemma', choices=['openemma', 'direct'])
    ap.add_argument('--view', default='nusc', choices=['nusc', 'orig'],
                    help='nusc: virtual nuScenes CAM_FRONT 1600x900; orig: original CAM_5 2880x1860')
    ap.add_argument('--max-pixels', type=int, default=1600 * 900,
                    help='pre-resize cap (the nusc view is exactly 1600x900 so it is untouched)')
    ap.add_argument('--desc-tokens', type=int, default=128, help='OpenEMMA uses 128 for every Qwen call')
    ap.add_argument('--motion-tokens', type=int, default=256)
    ap.add_argument('--no-sys', action='store_true', help='drop the motion system message (OpenEMMA Qwen path)')
    args = ap.parse_args()

    import json
    from qwen_vl import chat_text, generate, load_qwen25vl
    db = ParkingDataset(args.dataroot, args.scenes, load_agents=False)
    infos = {x['token']: x for x in json.load(open(NUSC_VIRTUAL_INFOS))['infos']} if args.view == 'nusc' else None
    toks = iterate(args, db)
    sys_msg = None if args.no_sys else SYS_MOTION + SYS_ADDITION
    info = {
        'model': args.model, 'dtype': args.dtype, 'mode': args.mode, 'view': args.view,
        'max_pixels': args.max_pixels, 'prompt_source': 'OpenEMMA main.py @8403ea6 (Qwen path)',
        'system_message': sys_msg, 'history': '10 samples t-4.5..t0 s, [signed speed m/s, curvature*100]',
        'output': '10 [speed, curvature] pairs at 0.5 s -> trapezoid integration from current state',
        'horizon_s': N_FUT * DT, 'K': 1, 'privileged_inputs': 'none (no command)',
        'fallback': 'const_vel from causal ego_state when < %d pairs parsed' % MIN_PAIRS,
        'every': args.every, 'n_frames_requested': len(toks),
    }
    W = PredWriter(args, args.variant, info)
    todo = [t for t in toks if not W.done(t)]
    print('frames: %d requested, %d to do' % (len(toks), len(todo)))
    if not todo:
        W.save()
        return
    t_load = time.time()
    model, proc = load_qwen25vl(args.model, args.dtype)
    print('model loaded in %.0fs; GPU mem %.1f GB' % (time.time() - t_load, torch.cuda.memory_allocated() / 1e9))
    t_fut = np.round(np.arange(1, N_FUT + 1) * DT, 3)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def user_msg(text):
        return [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]

    for bi, batch in enumerate(batched(todo, args.batch_size)):
        batch = [t for t in batch if not W.done(t)]  # another worker may have done them
        if not batch:
            continue
        ims, paths = zip(*[image_for(t, args.view, db, infos, args.max_pixels) for t in batch])
        descs = {t: {} for t in batch}
        if args.mode == 'openemma':
            texts, images, keys = [], [], []
            for t, im in zip(batch, ims):
                for name, p in (('scene', P_SCENE), ('objects', P_OBJECTS), ('intent', P_INTENT)):
                    texts.append(chat_text(proc, user_msg(p)))
                    images.append([im])
                    keys.append((t, name))
            outs, _ = generate(model, proc, texts, images, args.desc_tokens)
            for (t, name), o in zip(keys, outs):
                descs[t][name] = o
        texts, images, states = [], [], []
        for t, im in zip(batch, ims):
            hist, st0 = history_string(db, t)
            states.append((hist, st0))
            if args.mode == 'openemma':
                prompt = MOTION_OPENEMMA.format(scene=descs[t]['scene'], objects=descs[t]['objects'],
                                                intent=descs[t]['intent'], hist=hist)
            else:
                prompt = MOTION_DIRECT.format(hist=hist)
            msgs = ([{'role': 'system', 'content': [{'type': 'text', 'text': sys_msg}]}] if sys_msg else []) + user_msg(prompt)
            texts.append(chat_text(proc, msgs))
            images.append([im])
        outs, _ = generate(model, proc, texts, images, args.motion_tokens)
        # one sampled retry for unparseable outputs (OpenEMMA retries up to 3x)
        retry = {}
        bad = [i for i, o in enumerate(outs) if len(parse_pairs(o)) < MIN_PAIRS]
        if bad:
            torch.manual_seed(1234 + bi)
            routs, _ = generate(model, proc, [texts[i] for i in bad], [images[i] for i in bad], args.motion_tokens,
                                dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=50))
            for i, o in zip(bad, routs):
                retry[i] = o
        for i, t in enumerate(batch):
            hist, st0 = states[i]
            text = outs[i]
            pairs = parse_pairs(text)
            used_retry = False
            if len(pairs) < MIN_PAIRS and i in retry and len(parse_pairs(retry[i])) >= MIN_PAIRS:
                text, pairs, used_retry = retry[i], parse_pairs(retry[i]), True
            extra = {'motion_text': text, 'history': hist, 'image': paths[i], 'retry_used': used_retry}
            if i in retry and not used_retry:
                extra['retry_text'] = retry[i]
            extra.update({k: v for k, v in descs[t].items()})
            all_text = ' '.join([text] + list(descs[t].values()))
            extra['parking_vocab'] = parking_vocab(all_text)
            if len(pairs) >= MIN_PAIRS:
                n = len(pairs)
                pairs = (pairs + [pairs[-1]] * N_FUT)[:N_FUT]
                v = np.array([p[0] for p in pairs])
                k = np.array([p[1] for p in pairs]) / 100.0
                xy = integrate_speed_curvature(v, k, DT, v0=st0['speed'], k0=curvature_from_state(st0))
                extra.update(parse_fail=False, n_pairs=n, speed_curv=pairs)
                if not np.all(np.isfinite(xy)):
                    xy = const_vel_fallback(db, t, t_fut)
                    extra.update(parse_fail=True, fallback='const_vel', reason='non-finite')
            else:
                xy = const_vel_fallback(db, t, t_fut)
                extra.update(parse_fail=True, fallback='const_vel', n_pairs=len(pairs))
            W.add(t, t_fut, xy[None], extra=extra,
                  raw={'motion': text, **descs[t], 'retry': retry.get(i)})
        done = (bi + 1) * args.batch_size
        if bi % 5 == 0:
            print('[%s] %s  peak mem %.1f GB' % (args.variant, eta(min(done, len(todo)), len(todo), t0),
                                                  torch.cuda.max_memory_allocated() / 1e9), flush=True)
    W.info['runtime_s_per_frame'] = (time.time() - t0) / len(todo)
    W.info['peak_gpu_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
    W.save()
    print('stats', W.stats())


if __name__ == '__main__':
    main()
