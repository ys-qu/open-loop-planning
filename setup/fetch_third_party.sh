#!/usr/bin/env bash
# Clone the third-party repositories used by the adapters into third_party/, at the exact
# commits that were tested (~1.5 GB). No third-party file is modified afterwards.
#   bash setup/fetch_third_party.sh            # all repos
#   bash setup/fetch_third_party.sh UniAD VAD  # only some
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"

# dir                    url                                                     commit
REPOS="
VAD                      https://github.com/hustvl/VAD.git                        1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8
UniAD                    https://github.com/OpenDriveLab/UniAD.git                609ee083ea51c3521c323f1279dfc4cee0e60467
navsim                   https://github.com/autonomousvision/navsim.git           0a380a9063d7162ec93d0f51e9990ebac585f720
DiffusionDrive_navsim    https://github.com/hustvl/DiffusionDrive.git             9b52ed0ec06b073d82d6f392ab084c7b301c8681
GTRS                     https://github.com/NVlabs/GTRS.git                       92a740def80610e4096962d25cc837b21e72ef78
AutoVLA                  https://github.com/ucla-mobility/AutoVLA.git             ba34eed74ce6729e7986592d0e66cbaca397b4fa
Impromptu-VLA            https://github.com/ahydchh/Impromptu-VLA.git             84f0f8516ea78f6dc0325c373fd90b7fdb07d029
"

want=" $* "
echo "$REPOS" | while read -r dir url commit; do
  [ -z "$dir" ] && continue
  if [ $# -gt 0 ] && [[ "$want" != *" $dir "* ]]; then continue; fi
  if [ ! -d "$dir/.git" ]; then
    echo "[fetch] cloning $dir"
    git clone -q "$url" "$dir"
  fi
  git -C "$dir" fetch -q origin "$commit" 2>/dev/null || true
  git -C "$dir" -c advice.detachedHead=false checkout -q "$commit"
  echo "[fetch] $dir @ $(git -C "$dir" rev-parse --short HEAD)"
done
