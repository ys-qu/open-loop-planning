#!/usr/bin/env bash
# Positive controls on the real NAVSIM mini split (see navsim_mini_check.py), GPU 1 with the lock.
#   bash methods/navsim/run_mini_checks_gpu.sh transfuser hydra_mdp
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$(conda info --base)/envs/olp_navsim/bin/python}
CUDA_VISIBLE_DEVICES=${GPU:-0} flock locks/gpu${GPU:-0}.lock bash -c "
for a in $*; do
  $PY methods/navsim/navsim_mini_check.py --agent \$a --n 120 > logs/navsim_minicheck_\${a}.log 2>&1
  echo \"[\$(date)] mini check \$a done (exit \$?)\"
done
"
