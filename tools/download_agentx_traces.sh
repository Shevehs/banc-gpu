#!/usr/bin/env bash
# Downloads the AgentX traces (coding-agent sessions published by SemiAnalysis, Apache-2.0 license according to
# the Hugging Face dataset card; 393 traces, about 1.85 GB) into a local folder, for agentx.sh without
# Internet access (AGENTX_DIR=<folder>). Do this once, on a machine with Hugging Face access.
# Usage: ./tools/download_agentx_traces.sh [folder] [repo]
# Requires the .venv-aiperf environment to be activated (the hf command comes with AIPerf).
set -euo pipefail
TARGET="${1:-traces-agentx}"
REPO="${2:-semianalysisai/cc-traces-weka-062126}"
hf download "$REPO" --repo-type dataset --local-dir "$TARGET"
N=$(find "$TARGET" -name '*.json' -not -path '*/.cache/*' | wc -l | tr -d ' ')
echo "$N JSON file(s) in $TARGET."
echo "agentx.sh reads the *.json files of a single folder: if the traces are in a subfolder, give that"
echo "subfolder in AGENTX_DIR. Check the license on the dataset card before copying it elsewhere."
