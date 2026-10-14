#!/usr/bin/env bash
# Download the released checkpoints of the recommended planners (all public, no HF token needed).
#   bash setup/download_ckpts.sh                 # everything (~37 GB in ckpts/ + ~24 GB in the HF cache)
#   bash setup/download_ckpts.sh uniad navsim    # only some groups (see ALL_GROUPS)
#   bash setup/download_ckpts.sh --check         # only check that every source is reachable
# Needs `gdown` and the Hugging Face CLI on PATH:  pip install -U gdown huggingface_hub
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CK=$ROOT/ckpts
CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi
ALL_GROUPS="vad uniad navsim diffusiondrive gtrs qwen impromptu autovla"
if [ $# -eq 0 ] || [ "$*" = "all" ] || [ "$*" = "recommended" ]; then set -- $ALL_GROUPS; fi

# huggingface_hub >= 1.0 ships `hf`; older versions ship `huggingface-cli` (same `download`
# syntax, except that `hf` wants one --include/--exclude flag per pattern - used that way below)
if command -v hf >/dev/null 2>&1; then HFCLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HFCLI=huggingface-cli
else HFCLI=""; fi
need() { [ $CHECK = 1 ] && return 0; command -v "$1" >/dev/null 2>&1 || { echo "missing '$1': pip install -U gdown huggingface_hub"; exit 1; }; }
hfcli() { [ -n "$HFCLI" ] || need hf; "$HFCLI" "$@"; }

ok=0; bad=0
probe() {  # probe <label> <url>
  code=$(curl -s -o /dev/null -w '%{http_code}' -L -r 0-0 --max-time 60 "$2" || echo 000)
  if [[ "$code" =~ ^(200|206|302|303)$ ]]; then echo "[check] OK   $1"; ok=$((ok+1)); else echo "[check] FAIL $1 (HTTP $code) $2"; bad=$((bad+1)); fi
}
hf_files() {  # hf_files <repo> <local_dir> <file...>
  local repo=$1 dir=$2; shift 2
  if [ $CHECK = 1 ]; then for f in "$@"; do probe "$repo/$f" "https://huggingface.co/$repo/resolve/main/$f"; done; return; fi
  hfcli download "$repo" "$@" --local-dir "$dir"
}
url_file() {  # url_file <url> <out>
  if [ $CHECK = 1 ]; then probe "$(basename "$2")" "$1"; return; fi
  mkdir -p "$(dirname "$2")"; [ -s "$2" ] || curl -L --fail -C - -o "$2" "$1"
}
gd_file() {  # gd_file <google drive file id> <out>
  if [ $CHECK = 1 ]; then probe "gdrive:$(basename "$2")" "https://drive.google.com/uc?export=download&id=$1"; return; fi
  need gdown; mkdir -p "$(dirname "$2")"; [ -s "$2" ] || gdown "$1" -O "$2"
}

dl() {
  case $1 in
  vad)   # VAD-Base (hustvl/VAD README)
    gd_file 1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS "$CK/vad/VAD_base.pth" ;;
  uniad) # UniAD 2.0 base e2e + motion anchors (the UniAD config expects the anchors under its data/ dir)
    url_file https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_e2e.pth "$CK/uniad/uniad_base_e2e.pth"
    url_file https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/data/motion_anchor_infos_mode6.pkl \
             "$CK/uniad/motion_anchor_infos_mode6.pkl"
    if [ $CHECK = 0 ]; then
      [ -d "$ROOT/third_party/UniAD" ] || { echo "run setup/fetch_third_party.sh first (UniAD anchors link)"; exit 1; }
      mkdir -p "$ROOT/third_party/UniAD/data/others"
      ln -sf "$CK/uniad/motion_anchor_infos_mode6.pkl" "$ROOT/third_party/UniAD/data/others/motion_anchor_infos_mode6.pkl"
    fi ;;
  navsim) # NAVSIM baselines (TransFuser, Ego-status MLP); TransFuser builds timm resnet34 with
          # pretrained=True before loading the checkpoint, so pre-fetch it for offline machines
    hf_files autonomousvision/navsim_baselines "$CK/navsim_baselines" \
             transfuser/transfuser_seed_0.ckpt ego_status_mlp/ego_status_mlp_seed_0.ckpt
    if [ $CHECK = 1 ]; then probe timm/resnet34.a1_in1k https://huggingface.co/timm/resnet34.a1_in1k/resolve/main/config.json
    else hfcli download timm/resnet34.a1_in1k; fi ;;
  diffusiondrive) # DiffusionDrive (NAVSIM) + its trajectory anchors
    hf_files hustvl/DiffusionDrive "$CK/diffusiondrive" diffusiondrive_navsim_88p1_PDMS
    url_file https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy \
             "$CK/diffusiondrive/kmeans_navsim_traj_20.npy" ;;
  gtrs)  # Hydra-MDP (image-only, V2-99) from the GTRS release; dd3d_det_final.pth = backbone init used at build time
    hf_files Zzxxxxxxxx/gtrs "$CK/gtrs" hydra_mdp_vov.ckpt dd3d_det_final.pth ;;
  qwen)  # HF cache: Qwen2.5-VL-7B (OpenEMMA-style planner), Qwen2.5-VL-3B (AutoVLA base architecture/processor)
    for m in Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-3B-Instruct; do
      if [ $CHECK = 1 ]; then probe $m https://huggingface.co/$m/resolve/main/config.json; else hfcli download $m; fi
    done ;;
  impromptu) # Impromptu-VLA 7B, nuScenes-planning fine-tune (skip DeepSpeed optimizer / rng states)
    if [ $CHECK = 1 ]; then probe Impromptu-VLA-7B_AD https://huggingface.co/aaaaaap/ImpromptuVLAModel/resolve/main/7B_AD_finetune/config.json
    else
      if [ "$HFCLI" = hf ]; then EXC=(--exclude "7B_AD_finetune/global_step*" --exclude "7B_AD_finetune/rng_state_*")
      else EXC=(--exclude "7B_AD_finetune/global_step*" "7B_AD_finetune/rng_state_*"); fi   # huggingface-cli: one flag, many values
      hfcli download aaaaaap/ImpromptuVLAModel --include "7B_AD_finetune/*" "${EXC[@]}" --local-dir "$CK/impromptu_vla"
    fi ;;
  autovla) # AutoVLA (NAVSIM, PDMS 89; LoRA merged) - also needs the qwen group (3B base)
    hf_files Zewei-Zhou/AutoVLA "$CK/autovla" AutoVLA_PDMS_89.ckpt ;;
  *) echo "unknown group $1 (groups: $ALL_GROUPS)"; exit 1 ;;
  esac
}

for g in "$@"; do dl "$g"; done
if [ $CHECK = 1 ]; then echo "[check] $ok reachable, $bad failed"; [ $bad = 0 ]; fi
