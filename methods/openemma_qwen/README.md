# OpenEMMA-style training-free planning with Qwen2.5-VL (zero-shot)

> **Setup (shared with all methods)**, from the repo root after `source env.sh`:
> - code: `bash setup/fetch_third_party.sh`
> - env: `bash setup/create_env.sh olp_vlm`; `methods/openemma_qwen/setup_env.sh` does the same
> - weights: `bash setup/download_ckpts.sh qwen`
> - derived inputs: top-level README, "Run a single method"
>
> Paths below are relative to the repo root. `$OLP_DERIVED` and `$CONDA_BASE` are
> set by `env.sh`.

Status: **works** - all 960 keyframes, 0.1 % parse failures (numbers at the bottom are sanity checks only).

| | |
|---|---|
| Paper | OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving (Xing et al., arXiv 2412.15208) |
| Repo / commit | https://github.com/taco-group/OpenEMMA @ `8403ea6` (`third_party/OpenEMMA`, only used as the prompt reference) |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` (public HF, default cache `~/.cache/huggingface/hub`); optional `Qwen/Qwen2.5-VL-3B-Instruct` |
| Env | `olp_vlm` (`bash methods/openemma_qwen/setup_env.sh` -> `methods/vlm_common/setup_env_olp_vlm.sh`) |
| Code | `methods/openemma_qwen/infer.py` (+ `methods/vlm_common/`) |
| Output | `outputs/openemma_qwen25vl7b/predictions.json` (+ `raw_outputs.jsonl`) |

## Setup
```bash
bash methods/openemma_qwen/setup_env.sh                 # conda env olp_vlm
bash setup/download_ckpts.sh qwen                        # Qwen2.5-VL-7B/3B-Instruct -> HF cache
```

## Run
```bash
bash methods/openemma_qwen/run.sh                        # = the command below (resumable)
CUDA_VISIBLE_DEVICES=0 flock locks/gpu0.lock methods/vlm_common/run_in_env.sh olp_vlm \
  python methods/openemma_qwen/infer.py --model Qwen/Qwen2.5-VL-7B-Instruct --view nusc \
  --mode openemma --batch-size 2 --variant openemma_qwen25vl7b   # batch 2 = 6 description sequences (fits 22 GB)
# subset first:   ... --every 4      (deterministic: every 4th keyframe in time order)
# sanity check:   python tools/eval_planning.py --pred outputs/openemma_qwen25vl7b/predictions.json --horizons 1 2 3
```
**Not bit-reproducible.** Two runs on the same 80 keyframes, same env and GPU, gave different
plans on 6 of them: the long fp16 greedy generations flip at near-tied tokens. ADE@3s varied by
about 0.03 m (`docs/notes.md` §7.5).

Options (implemented, **not run** for lack of GPU time): `--mode direct` (OpenEMMA's non-"openemma"
baseline prompt, no descriptions), `--view orig` (original CAM_5 2880x1860 pre-resized to <= 1600x900 px),
`--model Qwen/Qwen2.5-VL-3B-Instruct`, `--dtype fp16|fp16-vit32|fp32`.

## Pipeline (OpenEMMA `main.py`, method `openemma`, Qwen code path)
Per keyframe, on the current front image only (OpenEMMA's Qwen path also passes one image):
1. Three free-text calls, prompts verbatim (see `infer.py`: `P_SCENE`, `P_OBJECTS`, `P_INTENT`),
   `max_new_tokens=128` as in OpenEMMA (texts are often cut at 128 tokens, also in OpenEMMA).
2. Motion call (prompt `MOTION_OPENEMMA`, verbatim) with the three texts and the 5 s history
   `[v, k]` x 10 (t-4.5 ... t0 s, 0.5 s steps), formatted `"[%.1f,%.1f]"` like OpenEMMA.
   The model answers 10 future `[speed, curvature]` pairs (0.5 s steps = 5 s).
3. The pairs are integrated (trapezoid, dt = 0.5 s) starting from the current causal state.
Decoding: Qwen's default `generation_config` (top_k=1 -> greedy, repetition_penalty 1.05) as in OpenEMMA;
if fewer than 6 pairs can be parsed, one sampled retry (T=0.7, top_p 0.9); if that also fails ->
constant-velocity fallback (`extra.parse_fail=true`).

Description prompts (user turn = front image + text, OpenEMMA verbatim):
```
[scene]   You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings.
[objects] You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you.
[intent]  You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?
```

Exact motion prompt (system + user; `{...}` filled per frame):
```
[system] You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.
The speed v is in m/s; a negative v means the vehicle is moving backwards (reversing).
[user] <image>These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The scene is described as follows: {scene}. 
        The identified critical objects are {objects}. 
        The car's intent is {intent}. 
        The 5 second historical velocities and curvatures of the ego car are {hist}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:
```

## Input adaptation / deviations from OpenEMMA (all deliberate, documented)
* Image: the **virtual nuScenes CAM_FRONT** (`$OLP_DERIVED/nusc_virtual`,
  1600x900, nuScenes intrinsics, built by `tools/make_nusc_virtual.py`) because OpenEMMA was designed
  for nuScenes CAM_FRONT; full resolution (1596x896 after Qwen's 28-px rounding, 1824 visual tokens).
  `--view orig` gives the raw 100-deg CAM_5 instead.
* History: OpenEMMA uses per-step *displacements* of nuScenes ego poses as "speed" and 3-point
  curvature from positions; here speed = causal signed speed in m/s and curvature = yaw-rate / speed
  (1/m, 0 below 0.3 m/s) x 100, both from `methods/vlm_common/olp_inputs.past_states` (same estimator
  as `ParkingDataset.ego_state`). Signed speed + one added system sentence (units, negative = reversing) so
  that the model *can* express reversing.
* System message: OpenEMMA's Qwen path silently drops it (it is only sent to GPT); we send it as a
  system turn (`--no-sys` reproduces the drop).
* Integration: OpenEMMA integrates with `t = linspace(0, 10, 10)` (dt = 1.11 steps, first point =
  current pose); we integrate with dt = 0.5 s from the current state, so point k is at t = 0.5 k s.
* Intent prompt: OpenEMMA updates the intent statefully from the previous frame; we always use the
  stateless first-frame prompt (frames are independent; subsets give the same result as full runs).
* Motion call `max_new_tokens=256` (OpenEMMA 128) to reduce truncated answers.
* No navigation command (OpenEMMA has none) -> no privileged input at all.
* fp16 LLM + fp32 vision tower (`--dtype fp16-vit32`) on the Turing GPUs (bf16 unsupported).

## Output
`t` = 0.5 ... 5.0 s (10 points), K = 1, ego frame x forward / y left (the integration starts at the
ego origin with heading 0, so no frame conversion is needed).
`extra`: `scene`, `objects`, `intent`, `motion_text` (raw texts), `history` (prompt string),
`speed_curv` (parsed pairs, padded to 10 by holding the last pair if 6..9 were given), `n_pairs`,
`retry_used`, `parse_fail`, `parking_vocab`, `image`.

## Results (sanity check, `tools/eval_planning.py`)
**openemma_qwen25vl7b**
```
=== openemma_qwen25vl7b  (outputs/openemma_qwen25vl7b/predictions.json)  frames=960  native horizon=5.0s  K=[1]
subset                       n   ADE/FDE@1   ADE/FDE@2   ADE/FDE@3   ADE/FDE@nat    minADE@nat    col%  colGT%  revRec%
all                        960  0.08/ 0.17  0.19/ 0.43  0.33/ 0.77    0.69/ 1.70          0.69    6.25    4.06     22.8
dir=FWD                    417  0.12/ 0.23  0.28/ 0.62  0.49/ 1.19    1.08/ 2.78          1.08    7.91    6.00      nan
dir=STP                    442  0.02/ 0.04  0.04/ 0.10  0.08/ 0.19    0.17/ 0.44          0.17    3.17    3.17      nan
dir=REV                    101  0.22/ 0.47  0.51/ 1.05  0.80/ 1.63    1.36/ 2.72          1.36   12.87    0.00     22.8
```
- parse failures: 1 / 960 (0.1 %) -> constant-velocity fallback
- model text mentions a parking term (park*, stall, spot, space, revers*, back up/out, pull in/out, garage, parking lot, maneuver, aisle) in 893 / 960 frames (93.0 %); on-road terms (lane, traffic light, stop sign, intersection, ...) in 960 (100.0 %)
- runtime 9.5 s/frame on a Quadro RTX 6000 (batch 2 frames = 6 description + 2 motion sequences; 4 VLM calls per frame), peak GPU memory 20.9 GB (PyTorch allocated; the card needs ~22 GB free)

Vocabulary per text field (`methods/vlm_common/text_stats.py`, 960 frames; on-road words are partly
*prompted*: the scene and intent prompts explicitly ask about traffic lights and lane markings):

| field | parking term | manoeuvre term (stall/spot/space/revers*/back up/pull in-out/maneuver/aisle) | on-road term |
|---|---|---|---|
| scene description | 92.1 % | 40.5 % | 99.8 % |
| critical objects | 40.3 % | 3.5 % | 29.6 % |
| intent | 66.6 % | 8.9 % | 100 % |
| motion answer | 0 % | 0 % | 0.1 % |
| any | 93.0 % | 48.1 % | 100 % |

Term counts (frames, any field): park* 889, parking lot 817, space 336, spot 118, maneuver 107, aisle 5,
back up/out 4, revers* 1 | lane 960, traffic light 946, lane change 536, intersection 165, crosswalk 138,
stop sign 18. The model recognises the parking lot ("navigating through a parking lot"), but the intent
answers follow the prompt's on-road structure, typically "1. **Lane Markings**: The ego car appears to be in
the leftmost lane of the intersection ..." - also on reversing frames. No intent answer mentions reversing
(0 / 960; revers* appears once in any field), although 101 frames are REV. 45 / 960 motion answers contain
negative speeds (made possible by the added system sentence); revRec 22.8 %.

Known issues
* OpenEMMA truncates every Qwen answer at 128 tokens (kept), so descriptions end mid-sentence.
* One frame (`2c8892a4...`) is a refusal ("The image does not provide enough context ...") also after the
  sampled retry -> constant-velocity fallback.
* Uses the full 1600x900 image (1824 visual tokens): the descriptions pass dominates the runtime.

