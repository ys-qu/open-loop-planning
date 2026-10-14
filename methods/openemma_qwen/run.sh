#!/usr/bin/env bash
# OpenEMMA-style Qwen2.5-VL planning on the parking dataset (GPU 0, wrapped in the GPU lock).
# Resumable: re-running continues from outputs/<variant>/predictions.json.
set -e
cd "$(dirname "$0")/../.."
R="methods/vlm_common/run_in_env.sh olp_vlm python methods/openemma_qwen/infer.py"
L="env CUDA_VISIBLE_DEVICES=${GPU:-0} flock locks/gpu${GPU:-0}.lock"
# main variant: Qwen2.5-VL-7B, virtual nuScenes CAM_FRONT, full OpenEMMA pipeline
$L $R --model Qwen/Qwen2.5-VL-7B-Instruct --view nusc --mode openemma --batch-size 2 \
      --variant openemma_qwen25vl7b "$@"
