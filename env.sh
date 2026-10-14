# Source this file from the repo root before running anything:   source env.sh
# Edit the two dataset paths below (or export them before sourcing).
export OLP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OLP_DATAROOT="${OLP_DATAROOT:-/path/to/dataset}"          # unzipped dataset root
export OLP_DERIVED="${OLP_DERIVED:-$OLP_REPO/derived}"                # derived inputs (renders, infos)
export CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
export PYTHONNOUSERSITE=1      # keep ~/.local site-packages out of the conda envs
[ -f "$OLP_REPO/env.local.sh" ] && source "$OLP_REPO/env.local.sh"   # optional machine-local overrides
if [ ! -d "$OLP_DATAROOT/v1.0-trainval" ]; then
  echo "[env.sh] WARNING: OLP_DATAROOT=$OLP_DATAROOT does not look like the dataset root (no v1.0-trainval/)" >&2
fi
