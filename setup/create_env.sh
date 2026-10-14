#!/usr/bin/env bash
# Create one of the conda envs used by the adapters, with the exact package versions that were
# tested (setup/envs/<env>.txt = pip list of the working env; torch / mmcv-full are installed first
# from their own wheel indexes, then everything else with --no-deps).
#
#   bash setup/create_env.sh olp_uniad    # VAD-Base, UniAD
#   bash setup/create_env.sh olp_navsim   # Ego-status MLP, TransFuser, DiffusionDrive, Hydra-MDP
#   bash setup/create_env.sh olp_vlm      # Qwen2.5-VL-7B (OpenEMMA-style), Impromptu-VLA
#   bash setup/create_env.sh olp_autovla  # AutoVLA
#
# ENVNAME=<other name> creates the env under another name. Needs conda (any distribution; python is
# taken from conda-forge, so the Anaconda channel terms do not apply) and an NVIDIA driver that
# supports CUDA 12.4 (the wheels bring their own CUDA runtime; no CUDA toolkit is needed).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=${1:?usage: create_env.sh <olp_uniad|olp_navsim|olp_vlm|olp_autovla>}
ENVNAME=${ENVNAME:-$NAME}
LOCK=$ROOT/setup/envs/$NAME.txt
[ -f "$LOCK" ] || { echo "no lock file $LOCK"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1          # ~/.local site-packages must not leak into the env
unset PYTHONPATH                   # nor a system PYTHONPATH (e.g. ROS)

PT=https://download.pytorch.org/whl
MM=https://download.openmmlab.com/mmcv/dist
MMCV=""
case $NAME in
  olp_uniad)      PY=3.9;  TORCH="torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url $PT/cu118"
                  MMCV="mmcv-full==1.7.2 -f $MM/cu118/torch2.0/index.html" ;;
  olp_navsim)     PY=3.9;  TORCH="torch==2.0.1 torchvision==0.15.2" ;;
  olp_vlm)        PY=3.10; TORCH="torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url $PT/cu124" ;;
  olp_autovla)    PY=3.10; TORCH="torch==2.4.0+cu121 torchvision==0.19.0+cu121 --extra-index-url $PT/cu121" ;;
  *) echo "unknown env $NAME"; exit 1 ;;
esac

if conda env list | awk '{print $1}' | grep -qx "$ENVNAME"; then
  echo "[create_env] $ENVNAME exists; installing into it"
else
  conda create -y -n "$ENVNAME" --override-channels -c conda-forge "python=$PY"
fi
conda activate "$ENVNAME"
python -m pip install -q "pip==23.3.1" "setuptools==65.5.1" "wheel"
pip install $TORCH
[ -n "$MMCV" ] && pip install $MMCV
pip install --no-deps -r "$LOCK"

# ---- env-specific post steps
case $NAME in
  olp_uniad)
    # mmdet3d 1.0.0rc6 refuses mmcv > 1.7.0, but 1.7.2 is the only prebuilt cu118/torch2.0 wheel
    F=$(python -c "import importlib.util as u; print(u.find_spec('mmdet3d').origin)")
    sed -i "s/mmcv_maximum_version = '1.7.0'/mmcv_maximum_version = '1.7.2'/" "$F" ;;
esac

python - <<'EOF'
import importlib, torch
print('[create_env] torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
for m in ['mmcv', 'mmdet', 'mmdet3d', 'transformers', 'nuplan', 'pytorch_lightning', 'diffusers', 'timm']:
    try:
        mod = importlib.import_module(m); print('[create_env]', m, getattr(mod, '__version__', 'ok'))
    except ImportError:
        pass
EOF
echo "[create_env] done: $ENVNAME"
