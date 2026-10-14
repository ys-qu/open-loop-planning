#!/usr/bin/env bash
# Environment for methods/autovla: conda env `olp_autovla` with the tested package versions.
# The other setup steps are shared (run from the repo root):
#   code:     bash setup/fetch_third_party.sh
#   weights:  bash setup/download_ckpts.sh qwen autovla
#   inputs:   see README.md "Run a single method" (derived camera renders / infos)
exec bash "$(dirname "$0")/../../setup/create_env.sh" olp_autovla "$@"
