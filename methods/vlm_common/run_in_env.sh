#!/usr/bin/env bash
# Run a command inside a conda env with a clean python path:
#   - PYTHONPATH unset (this machine exports ROS humble paths)
#   - PYTHONNOUSERSITE=1 (~/.local/lib/python3.x/site-packages would otherwise shadow env packages)
# usage: methods/vlm_common/run_in_env.sh <conda-env> <command...>
ENV=$1; shift
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec "$@"
