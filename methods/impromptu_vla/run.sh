#!/usr/bin/env bash
# Impromptu-VLA 7B (nuScenes-planning fine-tune) on $OLP_SCENES; extra args are passed to infer.py.
set -e
cd "$(dirname "$0")/../.."
env CUDA_VISIBLE_DEVICES=${GPU:-0} flock locks/gpu${GPU:-0}.lock \
  methods/vlm_common/run_in_env.sh olp_vlm python methods/impromptu_vla/infer.py \
  --ckpt ckpts/impromptu_vla/7B_AD_finetune --variant impromptu_7b_ad_ft --batch-size 8 "$@"
