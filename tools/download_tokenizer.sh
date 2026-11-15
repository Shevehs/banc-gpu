#!/usr/bin/env bash
# Downloads the tokenizer of the benchmarked model into tools/tokenizer/ (do this once, on a machine
# with Hugging Face access), so that it then serves as the default tokenizer for sweep.sh,
# including on machines without Internet access.
# Usage: ./tools/download_tokenizer.sh [Hugging Face repo]
# Requires the .venv-aiperf environment to be activated (the hf command comes with AIPerf).
set -euo pipefail
REPO="${1:-Qwen/Qwen3.8-Flash-Next-FP8}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="$HERE/tokenizer"
mkdir -p "$TARGET"
hf download "$REPO" --local-dir "$TARGET" \
  --include "tokenizer.json" --include "tokenizer_config.json" --include "vocab.json" \
  --include "merges.txt" --include "special_tokens_map.json" --include "LICENSE*"
rm -rf "$TARGET/.cache"
echo "Tokenizer of $REPO in $TARGET:"
ls -la "$TARGET"
[ -f "$TARGET/tokenizer.json" ] || { echo "WARNING: tokenizer.json missing; check the repo name."; exit 1; }
echo "Check the model license before committing these files (git add tools/tokenizer)."
