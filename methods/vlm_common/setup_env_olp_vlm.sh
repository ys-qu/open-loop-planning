#!/usr/bin/env bash
# conda env `olp_vlm` (tested versions); see setup/create_env.sh
exec bash "$(dirname "$0")/../../setup/create_env.sh" olp_vlm "$@"
