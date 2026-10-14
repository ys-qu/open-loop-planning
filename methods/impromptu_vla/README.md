# Impromptu-VLA (nuScenes planning checkpoints), zero-shot

> **Setup (shared with all methods)**, from the repo root after `source env.sh`:
> - code: `bash setup/fetch_third_party.sh`
> - env: `bash setup/create_env.sh olp_vlm`; `methods/impromptu_vla/setup_env.sh` does the same
> - weights: `bash setup/download_ckpts.sh impromptu (or impromptu7b)`
> - derived inputs: top-level README, "Run a single method"
>
> Paths below are relative to the repo root. `$OLP_DERIVED` and `$CONDA_BASE` are
> set by `env.sh`.

Status: **works** - the 7B nuScenes-planning checkpoint (`7B_AD_finetune`) on 960 keyframes, 0 parse failures (sanity numbers at the bottom).

| | |
|---|---|
| Paper | Impromptu VLA: Open Weights and Open Data for Driving Vision-Language-Action Models (Chi et al., arXiv 2505.23757) |
| Repo / commit | https://github.com/ahydchh/Impromptu-VLA @ `84f0f85` (`third_party/Impromptu-VLA`) |
| Checkpoint | HF `aaaaaap/ImpromptuVLAModel`, sub-folder `7B_AD_finetune` (Qwen2.5-VL-7B + Impromptu data + nuScenes planning) -> `ckpts/impromptu_vla/7B_AD_finetune/` (safetensors; the DeepSpeed `global_step*` optimizer states and rng states are not downloaded) |
| Env | `olp_vlm` (`bash setup/create_env.sh olp_vlm`); transformers 4.52.3 = the version in Impromptu's `environment.yaml` |
| Code | `methods/impromptu_vla/infer.py` |
| Output | `outputs/impromptu_7b_ad_ft/predictions.json` |

## Setup / run
```bash
source env.sh
bash setup/create_env.sh olp_vlm
bash setup/download_ckpts.sh impromptu                       # ~16 GB
methods/vlm_common/run_in_env.sh olp_vlm python methods/impromptu_vla/infer.py --selftest
#   -> rebuilds all 6019 prompts of the released nuscenes_test.json byte-for-byte from their numbers
GPU=0 bash methods/impromptu_vla/run.sh                      # resumable; = the command below
CUDA_VISIBLE_DEVICES=0 methods/vlm_common/run_in_env.sh olp_vlm \
  python methods/impromptu_vla/infer.py --ckpt ckpts/impromptu_vla/7B_AD_finetune \
  --variant impromptu_7b_ad_ft --batch-size 8
$CONDA_BASE/envs/olp_vlm/bin/python tools/eval_planning.py --pred outputs/impromptu_7b_ad_ft/predictions.json --horizons 1 2 3
```

## Input adaptation (replicates the nuScenes planning query of the released `nuscenes_test.json`)
* Image: virtual nuScenes **CAM_FRONT** (`nusc_virtual`, 1600x900, nuScenes intrinsics). As in
  Impromptu's inference (`train/inference_scripts/sglang_infer.py`, LLaMA-Factory
  `image_max_pixels=262144`) it is pre-resized to 682x384 and Qwen's processor makes it 672x392
  (336 visual tokens).
* Template: LLaMA-Factory `qwen2_vl` (system "You are a helpful assistant."), `<image>` placeholder
  replaced in place by `<|vision_start|><|image_pad|><|vision_end|>`.
* History: up to 3.0 s at 0.5 s (7 entries t-3.0 ... t-0.0). `[x, y]` = position of the (virtual)
  nuScenes LIDAR_TOP origin (0.94 m ahead of the rear axle) relative to its current position, in the
  current frame rotated to x forward / y left - exactly Impromptu's `ego_all_trajs` (LiDAR-frame
  positions, then `x = forward, y = -rightward`). Per entry "Acceleration X/Y" (body-frame causal
  acceleration), "Velocity" (signed body-frame vx), "Steering angle".
  - nuScenes takes these from the CAN bus; the dataset has no CAN, so they come from the causal pose
    fits (`olp_inputs.past_states`). The **steering angle is a proxy**: steering-wheel angle =
    15 x atan(2.588 m x curvature), curvature = yaw-rate / speed (0 below 0.3 m/s); 15 and 2.588 m
    (Renault Zoe wheelbase) were fitted on the CAN steering vs. path curvature of nuscenes_test.json
    (corr 0.98). Parking manoeuvres give values up to ~+-7 rad (nuScenes 1-99 % range: -3.6 .. 4.4).
  - nuScenes histories are shorter at the start of a scene; here the full 3 s is always given
    (the dataset poses cover the whole recording).
* Exact user prompt (one line, `\n` only at the end):
```
You are an autonomous driving agent. You have access to a front view camera image of a vehicle <image>. Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, given the vehicle's intent inferred from the images.Provided are the previous ego vehicle status recorded over the last 3.0 seconds (at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. The data is presented in the format [x, y]:.(t-3.0s) [x, y], Acceleration: X ax, Y ay m/s^2, Velocity: v m/s, Steering angle: s (positive: left turn, negative: right turn), (t-2.5s) ..., (t-0.0s) [0.0, 0.0], Acceleration: X ax, Y ay m/s^2, Velocity: v m/s, Steering angle: s (positive: left turn, negative: right turn)
```
  (numbers formatted like the generator: `float(np.round(x, 2))`, `-0.0 -> 0.0`.)
* No navigation command (the Impromptu nuScenes prompt has none) -> no privileged input.
* Decoding: greedy (Impromptu's sglang settings: temperature 0.01, top_k 1, repetition_penalty 1.0).
* Numerics: fp16 LLM + fp32 vision tower (`--dtype fp16-vit32`, bf16 is not available on Turing).
  The checkpoints' `preprocessor_config.json` names `Qwen2_5_VLImageProcessor` (transformers
  4.49.dev); the processor is rebuilt with `Qwen2VLImageProcessor` from the same values
  (verified identical token ids / pixel values vs. the base Qwen processor).

## Output
Model answer: `<PLANNING>... [x, y]: [x1, y1], ..., [x6, y6]</PLANNING>` = LiDAR-origin waypoints
at 0.5 ... 3.0 s (relative to the current LiDAR position, x fwd / y left) -> converted to the ego
origin with `olpcommon.geometry.ref_traj_to_ego(xy, (0.94, 0))`. `t` = 0.5 ... 3.0 s, K = 1.
`extra`: `text` (raw answer), `prompt`, `lidar_xy` (parsed LiDAR waypoints), `parse_fail`,
`parking_vocab`, `image`. Fewer than 6 parsed points -> constant-velocity fallback (`parse_fail`).

## Results (sanity check, `tools/eval_planning.py`)
| variant | training data | ADE/FDE@3s all | FWD | STP | REV | revRec% | parse-fail% | s/frame | peak GPU mem |
|---|---|---|---|---|---|---|---|---|---|
| impromptu_7b_ad_ft | 7B Base + Impromptu + nuScenes | 0.36/0.89 | 0.63/1.60 | 0.07/0.19 | 0.49/1.04 | 59.4 | 0 | 1.2 | 19.7 GB |
| const_vel (reference) | - | 0.28/0.67 | 0.49/1.19 | 0.06/0.17 | 0.32/0.73 | 94.1 | - | - | - |

(fp16 LLM + fp32 ViT, batch 8, Quadro RTX 6000.) Full `eval_planning.py` rows:

**impromptu_7b_ad_ft**
```
=== impromptu_7b_ad_ft  (outputs/impromptu_7b_ad_ft/predictions.json)  frames=960  native horizon=3.0s  K=[1]
subset                       n   ADE/FDE@1   ADE/FDE@2   ADE/FDE@3   ADE/FDE@nat    minADE@nat    col%  colGT%  revRec%
all                        960  0.08/ 0.15  0.19/ 0.45  0.36/ 0.89    0.36/ 0.89          0.36    6.46    4.06     59.4
dir=FWD                    417  0.13/ 0.26  0.34/ 0.80  0.63/ 1.60    0.63/ 1.60          0.63    8.15    6.00      nan
dir=STP                    442  0.01/ 0.02  0.03/ 0.08  0.07/ 0.19    0.07/ 0.19          0.07    3.39    3.17      nan
dir=REV                    101  0.14/ 0.26  0.30/ 0.64  0.49/ 1.04    0.49/ 1.04          0.49   12.87    0.00     59.4
```
- parse failures: 0 / 960 (0.0 %) -> constant-velocity fallback
- runtime 1.23 s/frame, peak GPU memory 19.7 GB (PyTorch allocated)

Observations / known issues
* The model only emits the `<PLANNING>... [x, y]: ...</PLANNING>` string (no reasoning text), so there is
  no vocabulary to analyse; 0 parse failures on 960 frames.
* It follows the motion history: on REV frames 59 % of the plans go backwards (the history carries the
  reverse motion), but the plans are mostly kinematic extrapolations: it does not beat constant velocity
  overall, on FWD or on REV frames at 3 s (STP is on par).
* During development the other released checkpoints were also run (3B and 7B, with and without the
  Impromptu data; outputs not included): all gave 0.34-0.36 m ADE@3s, so the Impromptu
  (unstructured-road) data does not help in parking lots.
* Steering angle is a kinematic proxy (no CAN): parking turns give up to +-7.5 rad (clipped), beyond
  the nuScenes range seen in training.
* 4 keyframes (scene-0063 c84fd14d, 0064 3cc5c86a, 0066 36857288, 0077 4c0ffee8) have an isolated
  0.12-0.28 m ego-pose jump at t0 while the ego is stationary before and after (dataset artefact): the
  history then shows a sudden 0.3-0.4 m/s motion and the models plan to accelerate.

