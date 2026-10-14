# Shared tooling for the VLM / VLA adapters

Used by `methods/{openemma_qwen,impromptu_vla,autovla}`.

| File | What |
|---|---|
| `setup_env_olp_vlm.sh` | Creates conda env `olp_vlm` (torch 2.5.1+cu124, transformers 4.52.3, qwen-vl-utils 0.0.11) via `setup/create_env.sh olp_vlm`. Used by the Qwen2.5-VL planner (OpenEMMA-style) and Impromptu-VLA. AutoVLA has its own env, `olp_autovla`. |
| `run_in_env.sh <env> <cmd>` | Runs a command in a conda env with `PYTHONPATH` unset and `PYTHONNOUSERSITE=1`. **Always use it.** Otherwise a system `PYTHONPATH` (e.g. ROS) or packages in `~/.local/lib/python3.x/site-packages` come *before* the env's site-packages. |
| `olp_inputs.py` | numpy-only, causal input builders (details below). |
| `runner.py` | Common CLI (`--scenes/--every/--offset/--limit/--tokens/--batch-size`) and a resumable writer (`$OLP_OUTPUTS/<variant>/predictions.json` + `raw_outputs.jsonl`). Stats (parse failures, parking vocabulary) are stored in `info.stats`. |
| `qwen_vl.py` | Qwen2.5-VL loading for Turing GPUs (transformers 4.52.3) and batched `generate` (details below). |
| `navsim_mini.py` | Real NAVSIM mini-split frames (`$NAVSIM_MINI_ROOT`) in the adapter input format. Used only to validate the NAVSIM-trained AutoVLA adapter (`--navsim-val N`). |
| `viz.py` | Front image with projected GT / prediction, plus a BEV panel, for sign and axis checks. |
| `text_stats.py` | Parse-failure rate and vocabulary per text field (details below). |
| `summarize.py`, `readme_results.py` | One-line sanity table per variant, and the markdown blocks pasted into the method READMEs. Both wrap `tools/eval_planning.py`. |

**`olp_inputs.py`** builds, from past data only:
- `past_states()`: ego pose plus velocity / acceleration / yaw rate at past times, with the same
  estimator as `ParkingDataset.ego_state`;
- curvature and a nuScenes steering-angle proxy;
- positions of a vehicle-fixed point (the virtual nuScenes LiDAR);
- the constant-velocity fallback and (speed, curvature) integration;
- the parking-vocabulary regexes.

**`qwen_vl.py`** loading modes:
- `fp16-vit32` (default): LLM in fp16, vision tower in fp32. Checked against fp32 on Impromptu-3B:
  5 of 8 answers identical, max waypoint difference 6 cm.
- `fp16` and `fp32`.

It applies three inference-only patches, all numerically equivalent:
1. **Vision attention.** SDPA builds a dense N×N mask over all patches of all images in a batch,
   which runs out of memory with 1600×900 images. It is replaced by per-segment attention
   (unit-tested against the masked version, max diff 5e-7).
2. **LM head.** At prefill it computes logits for every prompt position (several GB); only the last
   position is computed.
3. **ViT.** It runs on at most `VIT_CHUNK` images at a time, and identical images within a batch or
   in consecutive calls are encoded once (`VIT_CACHE=0` disables this; generations verified
   identical).

It also rebuilds the processor of Impromptu checkpoints, whose `preprocessor_config.json` names the
removed `Qwen2_5_VLImageProcessor`.

**`text_stats.py`** reports the share of frames whose text contains each kind of term, per field
(scene / objects / intent / text), overall and by ego direction:
- parking terms: park*, stall, spot, space, revers*, back up/out, pull in/out, garage, parking lot,
  maneuver, aisle;
- "manoeuvre terms": the subset stall/spot/space/revers*/back up/pull in-out/maneuver/aisle;
- on-road terms: lane, keep lane, traffic light, stop sign, intersection, lane change, speed bump,
  crosswalk, highway.

## Conventions shared by all adapters
* **Output:** `outputs/<variant>/predictions.json` (`olpcommon.save_predictions`), in the ego frame
  (x forward, y left), with `trajs[0]` = top-1.
  - `extra` holds the raw model text (`text`, `motion_text`, `scene`, `intent`, ...),
    `parse_fail` (bool), `parking_vocab` (terms found in the text) and method-specific fields.
* **Unparseable output:** replaced by a constant-velocity trajectory from `ParkingDataset.ego_state`
  (causal) and flagged `extra.parse_fail = true, extra.fallback = "const_vel"`. The rate is in
  `info.stats.parse_fail_pct`.
* **GPU jobs:** `CUDA_VISIBLE_DEVICES=$GPU flock locks/gpu$GPU.lock ...` (see each `run.sh`).
* **Sanity check:** `python tools/eval_planning.py --pred outputs/<variant>/predictions.json --horizons 1 2 3`
  and `methods/vlm_common/run_in_env.sh olp_vlm python methods/vlm_common/viz.py --pred ...`.

## Results (sanity check only, `tools/eval_planning.py`, scenes_subset.txt, 960 keyframes)

All runs are zero-shot with causal inputs; AutoVLA additionally gets the GT-derived command.
"Parking terms" and "on-road terms" are the share of frames whose model text contains such a
term (`text_stats.py`). Models that only emit trajectories have no text.

| variant | ADE/FDE@3s all | FWD | STP | REV | revRec% | parse-fail% | parking terms% | on-road terms% |
|---|---|---|---|---|---|---|---|---|
| openemma_qwen25vl7b | 0.33/0.77 | 0.49/1.19 | 0.08/0.19 | 0.80/1.63 | 22.8 | 0.1 | 93.0 | 100.0 |
| impromptu_7b_ad_ft | 0.36/0.89 | 0.63/1.60 | 0.07/0.19 | 0.49/1.04 | 59.4 | 0.0 | 0.0 | 0.0 |
| autovla_pdms89 | 0.93/2.90 | 0.95/2.75 | 0.65/2.60 | 2.09/4.82 | 0.0 | 0.0 | 0.0 | 0.0 |
| const_vel | 0.28/0.67 | 0.49/1.19 | 0.06/0.17 | 0.32/0.73 | 94.1 | 0.0 | - | - |

- The OpenEMMA-style "parking terms" come almost entirely from "parking lot" / "parked cars" in the
  scene description. Its intents read like on-road driving, and none of the 960 mentions reversing.
- AutoVLA never reasons on its own ("straightforward scenario" in 960/960 answers).

Method READMEs: `methods/openemma_qwen`, `methods/impromptu_vla`, `methods/autovla`.
