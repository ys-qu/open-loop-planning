#!/usr/bin/env python
"""AutoVLA (ucla-mobility/AutoVLA, NAVSIM PDMS-89 checkpoint) zero-shot on the parking dataset.

Re-implements AutoVLA.predict / get_prompt (third_party/AutoVLA/models/autovla.py, CoT
system prompt since the released checkpoint config `qwen2.5-vl-3B-nuplan-grpo-cot.yaml` has
use_cot: true) without the training-only dependencies (lightning, nuplan, navsim):
  * three videos of 4 frames at 2 Hz (t-1.5 .. t0): virtual NAVSIM CAM_F0, CAM_L0, CAM_R0
    (methods/navsim/olp_navsim renders, distorted 1920x1080), min/max_pixels = 109760;
  * text: |velocity| and |acceleration| (m/s, m/s^2, 3 decimals) and the driving instruction
    ('turn left' / 'keep forward' / 'turn right', NAVSIM VlaAgentFeatureBuilder mapping) from
    ParkingDataset.command() (privileged);
  * decoding: greedy (the released config samples with temperature 0.01, top_k 0, top_p 1);
  * action tokens (<action_i>, ids >= 151665) -> codebook rollout -> 10 poses (x, y, heading)
    at 0.5 s (5 s) in the NAVSIM rear-axle frame = dataset ego frame.
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AUTOVLA = os.path.join(ROOT, 'third_party', 'AutoVLA')
sys.path.insert(0, os.path.join(HERE, '..', 'vlm_common'))
sys.path.insert(0, os.path.join(ROOT, 'methods', 'navsim'))
import numpy as np  # noqa: E402

from olp_inputs import const_vel_fallback, parking_vocab  # noqa: E402
from runner import ParkingDataset, PredWriter, base_argparser, eta, iterate  # noqa: E402

CKPT = os.path.join(ROOT, 'ckpts', 'autovla', 'AutoVLA_PDMS_89.ckpt')
CONFIG = os.path.join(AUTOVLA, 'config', 'training', 'qwen2.5-vl-3B-nuplan-grpo-cot.yaml')
BASE = 'Qwen/Qwen2.5-VL-3B-Instruct'
CMD_TEXT = {'LEFT': 'TURN LEFT', 'STRAIGHT': 'KEEP FORWARD', 'RIGHT': 'TURN RIGHT', 'UNKNOWN': 'UNKNOWN'}
N_FUT, DT = 10, 0.5
SIDE_CAMS = {'l1r1': ('CAM_L1', 'CAM_R1'), 'l0r0': ('CAM_L0', 'CAM_R0')}
# assistant-turn prefix of AutoVLA's CoT samples (dataset_utils/sft_dataset.py)
FORCE_COT_PREFIX = "<think>\nThis is a complex scenario requiring additional reasoning.\n"
MIN_POSES = 6

SYS_COT = ("You are an Advanced Driver Assistance and Full Self-Driving System. "
           "You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. "
           "Your task is to predict the optimal driving action for the next five seconds.\n\n"
           "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
           "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
           "Structure your reasoning as follows:\n"
           "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
           "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
           "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
           "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
           "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
           "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
           "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
           "Present the final action clearly after your reasoning steps.")
SYS_NOCOT = ("You are an Advanced Driver Assistance and Full Self-Driving System. "
             "You will be provided with video observations from the ego vehicle’s surrounding cameras, along with the vehicle’s current dynamic states. "
             "Your task is to predict the most appropriate driving action for the next five seconds.")


def build_messages(cams, velocity, acceleration, instruction, min_pixels, max_pixels, use_cot=True):
    """= AutoVLA.get_prompt (models/autovla.py) with absolute image paths.
    cams: dict front_camera / front_left_camera / front_right_camera -> 4 paths (oldest first)."""
    def video(paths):
        return {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels,
                "video": ["file://%s" % p for p in paths]}
    user_content = [
        {"type": "text", "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."},
        {"type": "text", "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
        video(cams['front_camera']),
        {"type": "text", "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
        video(cams['front_left_camera']),
        {"type": "text", "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
        video(cams['front_right_camera']),
        {"type": "text", "text": (
            f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
            f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds.")},
    ]
    sys_text = SYS_COT if use_cot else SYS_NOCOT
    return [{"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {"role": "user", "content": user_content}]


class AutoVLARunner(object):
    def __init__(self, args, dev='cuda:0'):
        import torch
        import yaml
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        sys.path.insert(0, AUTOVLA)
        from models.action_tokenizer import ActionTokenizer
        self.torch, self.dev, self.args = torch, dev, args
        cfg = yaml.safe_load(open(CONFIG))
        cfg['model']['codebook_cache_path'] = os.path.join(AUTOVLA, cfg['model']['codebook_cache_path'])
        self.cfg = cfg
        # Build on the CPU in fp32 (base Qwen2.5-VL-3B + 2048 action tokens), load the AutoVLA
        # weights, cast (fp16-vit32: LLM fp16, vision tower fp32), then move to the GPU.
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32,
                                                                       attn_implementation='sdpa')
        self.processor = AutoProcessor.from_pretrained(args.base)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, model_config=cfg['model'])
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=False)
        sd = torch.load(args.ckpt, map_location='cpu', weights_only=True, mmap=True)['state_dict']
        sd = {k.replace('autovla.', ''): v for k, v in sd.items()}
        sd = {k[len('vlm.'):]: v for k, v in sd.items() if k.startswith('vlm.')}
        own = self.vlm.state_dict()
        miss = [k for k in own if k not in sd]
        unexp = [k for k in sd if k not in own]
        bad = [k for k in sd if k in own and sd[k].shape != own[k].shape]
        print('AutoVLA ckpt: %d tensors, missing %d %s, unexpected %d %s, shape-mismatch %d' % (
            len(sd), len(miss), miss[:3], len(unexp), unexp[:3], len(bad)))
        if miss or unexp or bad:
            raise RuntimeError('checkpoint does not match the model (transformers version?)')
        self.vlm.load_state_dict(sd, strict=True)
        del sd
        if args.dtype in ('fp16', 'fp16-vit32'):
            self.vlm.half()
            if args.dtype == 'fp16-vit32':
                self.vlm.visual.float()
        self.vlm.to(dev)
        self.vlm.eval()
        self.vdt = next(self.vlm.visual.parameters()).dtype
        self.action_start_id = cfg['model']['tokens']['action_start_id']
        self.vconf = cfg['model']['video']
        self.use_cot = cfg['model']['use_cot']

    def __call__(self, cams, velocity, acceleration, instruction):
        torch = self.torch
        from qwen_vl_utils import process_vision_info
        msgs = build_messages(cams, velocity, acceleration, instruction, self.vconf['min_pixels'], self.vconf['max_pixels'],
                              self.use_cot)
        image_inputs, video_inputs = process_vision_info(msgs)
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        if getattr(self.args, 'force_cot', False):
            text += FORCE_COT_PREFIX  # elicit the slow-thinking (CoT) mode of the SFT format
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
        inputs = inputs.to(self.dev)
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(self.vdt)
        with torch.no_grad():
            out = self.vlm.generate(**inputs, max_length=self.cfg['inference']['sample']['max_length'], do_sample=False,
                                    temperature=None, top_p=None, top_k=None)
        trimmed = out[0][inputs['input_ids'].shape[1]:].cpu()
        if len(trimmed) and int(trimmed[-1]) in (self.processor.tokenizer.eos_token_id, 151645, 151643):
            trimmed = trimmed[:-1]  # AutoVLA.predict drops the end token
        decoded = self.processor.decode(trimmed)
        if getattr(self.args, 'force_cot', False):
            decoded = FORCE_COT_PREFIX + decoded
        act = trimmed[trimmed >= self.action_start_id]
        traj = None
        if len(act):
            traj = self.action_tokenizer.decode_token_ids_to_trajectory(act)[0, 1:].numpy()
        meta = {'prompt_tokens': int(inputs['input_ids'].shape[1]), 'n_action_tokens': int(len(act)),
                'video_grid_thw': inputs['video_grid_thw'].tolist(), 'gen_tokens': int(len(trimmed))}
        return decoded, traj, meta


def strip_actions(text):
    return re.sub(r'(<action_\d+>)+', ' <ACTIONS> ', text).strip()


def navsim_validation(args):
    from navsim_mini import ade_fde, load_samples
    R = AutoVLARunner(args)
    names = {0: 'TURN LEFT', 1: 'KEEP FORWARD', 2: 'TURN RIGHT', 3: 'UNKNOWN'}
    res, cv = [], []
    for i, s in enumerate(load_samples(args.navsim_val, n_future=10)):
        L, Rr = SIDE_CAMS[args.side_cams]
        cams = {'front_camera': s['cams']['CAM_F0'], 'front_left_camera': s['cams'][L],
                'front_right_camera': s['cams'][Rr]}
        t0 = time.time()
        text, traj, meta = R(cams, float(np.hypot(*s['vel'])), float(np.hypot(*s['acc'])),
                             names[int(np.argmax(s['command']))].lower())
        dt = time.time() - t0
        res.append(ade_fde(traj[:8, :2], s['gt'][:8]) if traj is not None and len(traj) >= 8 else (np.nan, np.nan))
        cv.append(ade_fde(np.array(s['vel'])[None] * (np.arange(1, 9) * 0.5)[:, None], s['gt'][:8]))
        print('[%d] %.1fs ADE/FDE@4s %s  n_act=%d  %s' % (i, dt, np.round(res[-1], 2), meta['n_action_tokens'],
                                                         strip_actions(text)[:200].replace('\n', ' ')))
        if i == 0:
            print(meta)
            print('PRED', np.round(traj, 2).tolist() if traj is not None else None)
            print('GT  ', np.round(s['gt'], 2).tolist())
    r, c = np.array(res), np.array(cv)
    print('NAVSIM-mini AutoVLA n=%d ADE@4s %.2f FDE@4s %.2f (nan %d) | const-vel ADE %.2f FDE %.2f' % (
        len(r), np.nanmean(r[:, 0]), np.nanmean(r[:, 1]), int(np.isnan(r[:, 0]).sum()), c[:, 0].mean(), c[:, 1].mean()))


def main():
    ap = base_argparser(__doc__, 'autovla_pdms89')
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--base', default=BASE, help='HF id / path of Qwen2.5-VL-3B-Instruct (processor + architecture)')
    ap.add_argument('--dtype', default='fp16-vit32', choices=['fp32', 'fp16', 'fp16-vit32'])
    ap.add_argument('--navsim-val', type=int, default=0)
    ap.add_argument('--force-cot', action='store_true',
                    help='prefill the answer with the CoT opener of the SFT data ("<think> This is a complex scenario '
                         'requiring additional reasoning.") so the model writes its reasoning (separate variant)')
    ap.add_argument('--side-cams', default='l1r1', choices=list(SIDE_CAMS),
                    help="cameras of the 2nd/3rd video. l1r1 = released code for nuPlan (AutoVLAAgentFeatureBuilder "
                         "uses 'left/right_camera_paths' = NAVSIM cam_l1/cam_r1); l0r0 = front-left/right cameras")
    args = ap.parse_args()
    if args.navsim_val:
        navsim_validation(args)
        return
    import torch
    from olp_navsim import OLPNavsimInputs
    db = ParkingDataset(args.dataroot, args.scenes, load_agents=False)
    inp = OLPNavsimInputs(db)
    toks = iterate(args, db)
    info = {
        'checkpoint': os.path.abspath(args.ckpt), 'hf': 'Zewei-Zhou/AutoVLA/AutoVLA_PDMS_89.ckpt', 'base': args.base,
        'config': os.path.relpath(CONFIG, ROOT), 'dtype': args.dtype, 'use_cot': True, 'side_cams': args.side_cams,
        'force_cot': bool(args.force_cot),
        'inputs': 'virtual NAVSIM CAM_F0/L0/R0 x 4 frames (t-1.5..0, previous keyframes), |v|, |a| (ego_state), '
                  'instruction from ParkingDataset.command()',
        'privileged_inputs': 'driving command (ParkingDataset.command, GT future)', 'decoding': 'greedy, max_length 2048',
        'horizon_s': N_FUT * DT, 'K': 1, 'fallback': 'const_vel when < %d action tokens' % MIN_POSES,
    }
    W = PredWriter(args, args.variant, info)
    todo = [t for t in toks if not W.done(t)]
    print('frames: %d requested, %d to do' % (len(toks), len(todo)))
    if not todo:
        W.save()
        return
    R = AutoVLARunner(args)
    print('model loaded; GPU mem %.1f GB' % (torch.cuda.memory_allocated() / 1e9))
    t_fut = np.round(np.arange(1, N_FUT + 1) * DT, 3)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    L, Rr = SIDE_CAMS[args.side_cams]
    keys = {'front_camera': 'CAM_F0', 'front_left_camera': L, 'front_right_camera': Rr}
    for fi, t in enumerate(todo):
        if W.done(t):  # another worker may have done it
            continue
        htoks = inp.history_tokens(t)
        cams = {}
        for k, c in keys.items():
            ps = [inp.virtual_path(h, c) for h in htoks]
            missing = [p for p in ps if not os.path.exists(p)]
            if missing:
                raise RuntimeError('missing virtual NAVSIM renders (run methods/navsim/make_navsim_virtual.py): %s' % missing[:2])
            cams[k] = ps
        s = db.ego_state(t)
        cmd = db.command(t)
        instr = CMD_TEXT[cmd].lower()
        vel, acc = float(np.hypot(s['vx'], s['vy'])), float(np.hypot(s['ax'], s['ay']))
        text, traj, meta = R(cams, vel, acc, instr)
        extra = {'text': strip_actions(text), 'command': cmd, 'instruction': instr, 'velocity': vel,
                 'acceleration': acc, 'history_tokens': htoks, 'parking_vocab': parking_vocab(strip_actions(text)),
                 'n_action_tokens': meta['n_action_tokens'],
                 'cot': 'straightforward scenario' not in text.lower()}  # fast mode says "This is a straightforward scenario
        if traj is not None and len(traj) >= MIN_POSES:
            n = len(traj)
            if n < N_FUT:
                traj = np.concatenate([traj, np.repeat(traj[-1:], N_FUT - n, 0)])
            xy = traj[:N_FUT, :2]
            extra.update(parse_fail=False, poses=np.round(traj[:N_FUT], 3).tolist(), padded=n < N_FUT)
        else:
            xy = const_vel_fallback(db, t, t_fut)
            extra.update(parse_fail=True, fallback='const_vel')
        W.add(t, t_fut, xy[None], extra=extra, raw={'text': text})
        if fi == 0:
            print('META', meta)
            print('TEXT', text[:3000])
        if fi % 10 == 0:
            print('[%s] %s  peak mem %.1f GB  last: %d gen tokens' % (args.variant, eta(fi + 1, len(todo), t0),
                                                                     torch.cuda.max_memory_allocated() / 1e9, meta['gen_tokens']), flush=True)
    W.info['runtime_s_per_frame'] = (time.time() - t0) / len(todo)
    W.info['peak_gpu_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
    W.save()
    print('stats', W.stats())


if __name__ == '__main__':
    main()
