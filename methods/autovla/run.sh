#!/usr/bin/env bash
# AutoVLA (PDMS-89 checkpoint) on the parking dataset.
set -e
cd "$(dirname "$0")/../.."
env CUDA_VISIBLE_DEVICES=${GPU:-0} flock locks/gpu${GPU:-0}.lock \
  methods/vlm_common/run_in_env.sh olp_autovla python methods/autovla/infer.py --variant autovla_pdms89 "$@"
