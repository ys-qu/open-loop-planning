#!/usr/bin/env bash
# Run NAVSIM-family agents on all keyframes of the scene list on GPU 1 (one lock for the sequence).
#   bash methods/navsim/run_gpu.sh ego_mlp transfuser diffusiondrive hydra_mdp
# Output: outputs/navsim_<agent>/predictions.json, logs/navsim_<agent>_gpu.log
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$(conda info --base)/envs/olp_navsim/bin/python}
# frames used for the BEV figures (methods/navsim/viz_bev.py --print-tokens)
TOKS="b66f1894a3000c01d20539d86badd3fa ebdf899290774e5d1691e2fec314ccd3 bdd0f62500bd87b25d21d1a0707dcb47 22adc95b2d4c80419b5c1f84ffd19f1b a905be3fc27f1d7aac42c1d821d83cf3 fd1bb372b2685091e07656ce8593e936 46752df98fbe36141417836f2c41f02a"
AGENTS="$*"
CUDA_VISIBLE_DEVICES=${GPU:-0} flock locks/gpu${GPU:-0}.lock bash -c "
for a in $AGENTS; do
  echo \"[\$(date)] start \$a\"
  $PY methods/navsim/run_agent.py --agent \$a --variant navsim_\$a --batch-size 8 --workers 6 \
      --dump-aux logs/navsim_\${a}_aux.npz --aux-tokens $TOKS > logs/navsim_\${a}_gpu.log 2>&1
  echo \"[\$(date)] done \$a (exit \$?)\"
done
"
