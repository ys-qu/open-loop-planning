# AutoVLA (NAVSIM PDMS-89 checkpoint), zero-shot

> **Setup (shared with all methods)**, from the repo root after `source env.sh`:
> - code: `bash setup/fetch_third_party.sh`
> - env: `bash setup/create_env.sh olp_autovla`; `methods/autovla/setup_env.sh` does the same
> - weights: `bash setup/download_ckpts.sh autovla`
> - derived inputs: top-level README, "Run a single method"
>
> Paths below are relative to the repo root. `$OLP_DERIVED` and `$CONDA_BASE` are
> set by `env.sh`.

Status: **works** (sanity numbers at the bottom).

| | |
|---|---|
| Paper | AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning (Zhou et al., NeurIPS 2025, arXiv 2506.13757) |
| Repo / commit | https://github.com/ucla-mobility/AutoVLA @ `ba34eed` (`third_party/AutoVLA`) |
| Checkpoint | HF `Zewei-Zhou/AutoVLA`, file `AutoVLA_PDMS_89.ckpt` (16.3 GB Lightning ckpt, LoRA already merged) -> `ckpts/autovla/`; base architecture/processor `Qwen/Qwen2.5-VL-3B-Instruct` (HF cache); action codebook `third_party/AutoVLA/codebook_cache/agent_vocab.pkl` (2048 tokens) |
| Config | `third_party/AutoVLA/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml` (use_cot: true, video min/max_pixels 109760, inference: temperature 0.01, top_k 0, top_p 1, max_length 2048) |
| Env | `olp_autovla` (`bash methods/autovla/setup_env.sh`: torch 2.4.0+cu121, **transformers 4.49.0**, qwen-vl-utils 0.0.10) |
| Code | `methods/autovla/infer.py` (re-implements `models/autovla.py: AutoVLA.get_prompt/predict` without lightning / nuplan / navsim) |
| Output | `outputs/autovla_pdms89/predictions.json` (960 frames). An optional forced-reasoning mode (`--force-cot`) is described below; its output is not included. |

## Setup / run
```bash
bash methods/autovla/setup_env.sh
bash setup/download_ckpts.sh qwen autovla   # AutoVLA ckpt + Qwen2.5-VL-3B-Instruct (architecture + processor)
bash methods/autovla/run.sh                 # resumable; --every 2 for a subset
#   = CUDA_VISIBLE_DEVICES=0 flock locks/gpu0.lock methods/vlm_common/run_in_env.sh olp_autovla \
#       python methods/autovla/infer.py --variant autovla_pdms89
# extra variant with forced reasoning (every 4th keyframe):
CUDA_VISIBLE_DEVICES=0 flock locks/gpu0.lock methods/vlm_common/run_in_env.sh olp_autovla \
  python methods/autovla/infer.py --variant autovla_pdms89_forcedcot --force-cot --every 4
# adapter check on real NAVSIM frames:
CUDA_VISIBLE_DEVICES=0 flock locks/gpu0.lock methods/vlm_common/run_in_env.sh olp_autovla \
  python methods/autovla/infer.py --navsim-val 12
python tools/eval_planning.py --pred outputs/autovla_pdms89/predictions.json --horizons 1 2 3
```
Why transformers 4.49.0: the checkpoint's state-dict keys follow the 4.49 module layout and
AutoVLA loads with `strict=False`; with >= 4.50 the keys silently do not match. The adapter loads
with `strict=True` and checks key / shape agreement (825/825 tensors).

## Inputs (exactly the AutoVLA NAVSIM prompt)
* System prompt: the CoT system prompt of `AutoVLA.get_prompt` (scene analysis, critical objects,
  ..., lateral / longitudinal action sets).
* User: three "videos" of 4 frames at 2 Hz (t-1.5, -1.0, -0.5, 0 s) introduced as front,
  front-left and front-right views. **Camera choice:** for nuPlan the released feature builder
  (`navsim/navsim/agents/autovla_agent.py: AutoVLAAgentFeatureBuilder`) feeds
  `left_camera_paths` / `right_camera_paths` as the 2nd / 3rd video, and the released preprocessing
  (`dataset_utils/preprocessing/nuplan_dataset.py` + `vla_agent.py`) fills those keys with NAVSIM
  `cam_l1` / `cam_r1` (yaw ~ +-111 deg, side cameras), not `cam_l0` / `cam_r0` (+-55 deg). We follow
  the released code: CAM_F0 + CAM_L1 + CAM_R1 (`--side-cams l1r1`, default; `--side-cams l0r0` for the
  front-left/right cameras). Virtual nuPlan cameras (distorted 1920x1080) from
  `methods/navsim/olp_navsim` (CAM_L1 <- CAM_4, CAM_R1 <- CAM_3); history frames =
  previous keyframes. qwen-vl-utils resizes each frame to 420x224 (min = max = 109760 px):
  240 tokens per camera;
  text `The current velocity of the vehicle is {|v|:.3f} m/s, and the current acceleration is {|a|:.3f} m/s². The driving instruction is: {instruction}. ...`
* |v|, |a|: norms of the causal `ParkingDataset.ego_state` velocity / acceleration (AutoVLA uses norms, so
  the direction of motion is not given to the model).
* instruction: `turn left` / `keep forward` / `turn right` from `ParkingDataset.command()` (**privileged**;
  NAVSIM VlaAgent mapping of the one-hot command).
* Decoding: greedy (`do_sample=False`) instead of temperature-0.01 sampling (practically identical).
* Numerics: LLM fp16, vision tower fp32 (`--dtype fp16-vit32`); weights loaded in fp32 on the CPU, cast,
  then moved to the GPU.

Exact system prompt (`SYS_COT`, verbatim from `models/autovla.py`):
```
You are an Advanced Driver Assistance and Full Self-Driving System. You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. Your task is to predict the optimal driving action for the next five seconds.

First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.

If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.

Structure your reasoning as follows:
1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.
2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.
3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.
4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.
5. **Final Action Decision**: Select one lateral action and one longitudinal action:
- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]
- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]

Present the final action clearly after your reasoning steps.
```
User turn = the three text + video pairs above, then the velocity / acceleration / instruction sentence
(`build_messages` in `infer.py`); chat template of Qwen2.5-VL-3B-Instruct (transformers 4.49 does not
forward `add_vision_id`, as in AutoVLA's own runs).

## Output
The generated text = optional CoT + `<action_i>` tokens; action tokens (ids >= 151665) are decoded with
AutoVLA's `ActionTokenizer` (codebook rollout) to 10 poses (x, y, heading) at 0.5 s = 5 s, NAVSIM
rear-axle frame = dataset ego frame. `t` = 0.5 ... 5.0 s, K = 1. Fewer than 10 tokens: padded by holding
the last pose (flag `padded`); fewer than 6: constant-velocity fallback (`parse_fail`).
`extra`: `text` (CoT with the action tokens replaced by `<ACTIONS>`), `cot` (reasoning produced),
`n_action_tokens`, `poses`, `instruction`, `velocity`, `acceleration`, `history_tokens`, `parking_vocab`.

## Validation on real NAVSIM frames (`--navsim-val 12`)
Random moving frames (speed > 1 m/s, >= 5 s of future) of the local NAVSIM mini split, same code path:

| setting | n | ADE@4s | FDE@4s |
|---|---|---|---|
| AutoVLA, CAM_F0 + CAM_L1 + CAM_R1 (released nuPlan mapping, used) | 30 | 1.39 | 3.43 |
| AutoVLA, CAM_F0 + CAM_L0 + CAM_R0 | 30 | 1.69 | 4.11 |
| constant velocity | 30 | 1.81 | 4.37 |
| AutoVLA fp32 vs. fp16-vit32 | 12 | identical per-sample results | |

All 30 NAVSIM answers use the fast mode (`<think> This is a straightforward scenario, and a direct
decision can be made. </think>` + 10 action tokens). AutoVLA's L2 error is only moderately below
constant velocity in-domain too (the RL-tuned policy optimises PDMS, not L2), so the numbers on the
parking dataset should be read relative to this.

## Results (sanity check, `tools/eval_planning.py`)
**autovla_pdms89**
```
=== autovla_pdms89  (outputs/autovla_pdms89/predictions.json)  frames=960  native horizon=5.0s  K=[1]
subset                       n   ADE/FDE@1   ADE/FDE@2   ADE/FDE@3   ADE/FDE@nat    minADE@nat    col%  colGT%  revRec%
all                        960  0.13/ 0.26  0.39/ 1.07  0.93/ 2.90    2.95/ 9.21          2.95   12.19    4.06      0.0
dir=FWD                    417  0.14/ 0.30  0.43/ 1.17  0.95/ 2.75    2.66/ 7.76          2.66   12.95    6.00      nan
dir=STP                    442  0.02/ 0.04  0.16/ 0.63  0.65/ 2.60    2.89/10.18          2.89    7.69    3.17      nan
dir=REV                    101  0.59/ 1.09  1.24/ 2.62  2.09/ 4.82    4.41/10.97          4.41   28.71    0.00      0.0
```
- parse failures: 0 / 960 (0.0 %) -> constant-velocity fallback
- model text mentions a parking term (park*, stall, spot, space, revers*, back up/out, pull in/out, garage, parking lot, maneuver, aisle) in 0 / 960 frames (0.0 %); on-road terms (lane, traffic light, stop sign, intersection, ...) in 0 (0.0 %)
- runtime 2.9 s/frame (fast mode, ~44 generated tokens), peak GPU memory 10.1 GB (PyTorch allocated)

- runtime 17.6 s/frame (~400-500 generated tokens), peak GPU memory 10.1 GB

Notes
* `autovla_pdms89` (default decoding, all 960 frames): the model **never** switches to its reasoning
  mode on the parking dataset: 960/960 answers are `<think> This is a straightforward scenario, and a direct
  decision can be made. </think> <answer> The final output action is: <10 action tokens> </answer>`
  (same on NAVSIM mini), so its text contains no scene vocabulary at all (0 % parking / 0 % on-road).
  It never plans a reverse (revRec 0 %: |v| is given without sign and its action codebook rarely moves
  backwards) and it drives off when the ego is stationary (STP ADE@3s 0.65 m vs 0.06 m for constant
  velocity).
* Forced-reasoning mode (`--force-cot`; development run on every 4th keyframe = 240 frames, output not included): the
  answer is prefilled with the CoT opener of AutoVLA's SFT data (`<think>\nThis is a complex scenario
  requiring additional reasoning.\n`), so the model writes its scene analysis. The reasoning names the
  place ("parking lot" 235/240, parking term in 240/240, "maneuver" 112, "pull in/out" 34, "spot" 29,
  "space" 57) but uses on-road terms too (75.8 %: lane 136, traffic light 114, lane change 55,
  intersection 19, stop sign 15) and its final decision comes from the fixed on-road action set:
  "move forward with a constant speed" 112, "... with a deceleration" 64, "... deceleration to zero" 19,
  "... acceleration" 7, turn left/right 12 - reversing is never chosen (revers* in 7/240 texts).
  Plans are worse than in fast mode (ADE/FDE@3s 1.26/3.48 vs 0.95/2.94 m on the same 240 frames) and 1/240
  answer hit max_length (2048 tokens) before any action token -> constant-velocity fallback.

