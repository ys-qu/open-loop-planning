#!/usr/bin/env bash
# Run the recommended comparison set end to end on the scene list $OLP_SCENES:
#   derived inputs -> 9 planners + 3 kinematic baselines -> sanity scores (outputs/summary_recommended.{json,md}).
# One log per step in logs/run_<step>.log; a failing step is reported and the rest continues.
#
#   source env.sh && GPU=0 bash run_recommended.sh
#
# Full dataset: write a scene list (one scene name per line) and, before running,
#   export OLP_SCENES=$OLP_REPO/scenes_full.txt OLP_TAG=full OLP_OUTPUTS=$OLP_REPO/outputs_full
# Prerequisites: setup/fetch_third_party.sh, setup/create_env.sh for olp_uniad olp_navsim olp_vlm
# olp_autovla, setup/download_ckpts.sh recommended (see README.md, Setup).
set -uo pipefail
cd "$(dirname "$0")"
[ -n "${OLP_DATAROOT:-}" ] || source env.sh
G=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$G
LOCK="flock locks/gpu$G.lock"
E=${CONDA_BASE:-$(conda info --base)}/envs
TAG=${OLP_TAG:-subset}
OUT=${OLP_OUTPUTS:-$PWD/outputs}
mkdir -p logs locks "$OUT"
FAILED=""
step() {   # step <name> <command...>
  local name=$1; shift
  echo "[$(date '+%F %T')] $name"
  if "$@" > "logs/run_$name.log" 2>&1; then echo "    ok"; else echo "    FAILED -> logs/run_$name.log"; FAILED="$FAILED $name"; fi
}

# 1) derived inputs for the scene list (existing files are kept)
[ -f "${OLP_DERIVED}/nusc_virtual/infos_$TAG.json" ] || \
  step prep_nusc_virtual "$E/olp_uniad/bin/python" tools/make_nusc_virtual.py
step prep_navsim_virtual "$E/olp_navsim/bin/python" methods/navsim/make_navsim_virtual.py \
  --cams CAM_F0 CAM_L0 CAM_R0 CAM_B0 CAM_L1 CAM_R1 CAM_L2 CAM_R2 --workers 8

# 2) planners
step kinematic "$E/olp_uniad/bin/python" methods/kinematic/run.py           # stationary, const_vel, ctrv
for a in ego_mlp transfuser diffusiondrive hydra_mdp; do
  step "navsim_$a" $LOCK "$E/olp_navsim/bin/python" methods/navsim/run_agent.py --agent $a --variant navsim_$a \
       --batch-size 8 --workers 6
done
step vad_base $LOCK "$E/olp_uniad/bin/python" methods/vad/infer.py
step uniad    $LOCK "$E/olp_uniad/bin/python" methods/uniad/infer.py
step impromptu_7b_ad_ft $LOCK methods/vlm_common/run_in_env.sh olp_vlm python methods/impromptu_vla/infer.py \
     --ckpt ckpts/impromptu_vla/7B_AD_finetune --variant impromptu_7b_ad_ft --batch-size 8
step autovla  $LOCK methods/vlm_common/run_in_env.sh olp_autovla python methods/autovla/infer.py --variant autovla_pdms89
step openemma $LOCK methods/vlm_common/run_in_env.sh olp_vlm python methods/openemma_qwen/infer.py \
     --model Qwen/Qwen2.5-VL-7B-Instruct --view nusc --mode openemma --batch-size 2 --variant openemma_qwen25vl7b

# 3) sanity scores (same script and ground truth for every method)
V="stationary const_vel ctrv navsim_ego_mlp navsim_transfuser navsim_diffusiondrive navsim_hydra_mdp vad_base uniad_base impromptu_7b_ad_ft autovla_pdms89 openemma_qwen25vl7b"
P=""; for v in $V; do [ -f "$OUT/$v/predictions.json" ] && P="$P $OUT/$v/predictions.json"; done
step eval "$E/olp_uniad/bin/python" tools/eval_planning.py --pred $P --horizons 1 2 3 --by direction travel \
     --out "$OUT/summary_recommended.json"
"$E/olp_uniad/bin/python" tools/summarize.py "$OUT/summary_recommended.json" > "$OUT/summary_recommended.md" 2>/dev/null \
  && echo "table: $OUT/summary_recommended.md"
[ -z "$FAILED" ] && echo "all steps ok" || { echo "failed steps:$FAILED"; exit 1; }
