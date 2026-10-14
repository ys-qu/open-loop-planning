"""Machine-specific locations, all overridable by environment variables (see env.sh).

  OLP_DATAROOT  unzipped parking dataset (v1.0-trainval/, samples/, sweeps/, labels/, ...)
  OLP_DERIVED   derived inputs written by the data-prep scripts (virtual camera renders, infos);
               default <repo>/derived
Checkpoints live in <repo>/ckpts and third-party code in <repo>/third_party (symlink them
elsewhere if the disk is small).
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAROOT = os.environ.get('OLP_DATAROOT', '')
DERIVED = os.environ.get('OLP_DERIVED', os.path.join(REPO_ROOT, 'derived'))


def derived(*parts):
    return os.path.join(DERIVED, *parts)
