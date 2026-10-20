#!/usr/bin/env bash
# Builds the wrappers environment (plan.md, Environment, Phase A).
# Run from audio_codec_wrappers/:  bash scripts/install_env.sh
set -euo pipefail

cd "$(dirname "$0")/.."
VENV=../venv_audio_codec_wrappers
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"

# 1. torch, same build as venv_SCAPES (2.10.0+cu128)
$PIP install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# 2. pinned PyPI set
$PIP install -r requirements.txt
# 3. git dependencies, without their own dependency lists (D2)
$PIP install --no-deps -r requirements-git.txt
# 4. the package itself
$PIP install --no-deps -e .
# 5. the exact environment of the results
$PIP freeze > requirements.lock
