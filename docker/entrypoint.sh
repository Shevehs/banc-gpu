#!/usr/bin/env bash
# Start of the bench container (README, section 4.4): with BENCH_TOKENIZER_REPO set and no tokenizer in
# tools/tokenizer/ yet, download the model's tokenizer once; then run the command (the interface).
# A failed download is not fatal: the message says what to do, and the interface still starts.
set -u
if [ -n "${BENCH_TOKENIZER_REPO:-}" ] && [ ! -f tools/tokenizer/tokenizer.json ]; then
  echo "Tokenizer: downloading ${BENCH_TOKENIZER_REPO} into tools/tokenizer/ (once)"
  if ! ./tools/download_tokenizer.sh "$BENCH_TOKENIZER_REPO"; then
    echo "WARNING: the tokenizer download failed (no Internet access, a gated model without HF_TOKEN, or a wrong name)."
    echo "Copy tokenizer.json and tokenizer_config.json of the model into tools/tokenizer/ (README, section 5.3)."
  fi
fi
exec "$@"
