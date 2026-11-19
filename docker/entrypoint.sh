#!/usr/bin/env bash
# Start of the bench container (README, section 4.4):
# - with BENCH_TOKENIZER_REPO set and no complete tokenizer in tools/tokenizer/ yet, download the model's tokenizer
#   once (a failed download is not fatal: the message says what to do, and the interface still starts);
# - warn if the container runs as root on files owned by someone else (Linux: results would belong to root);
# - then run the command (the interface), without HF_TOKEN in its environment.
set -u
if [ -n "${BENCH_TOKENIZER_REPO:-}" ] && { [ ! -f tools/tokenizer/tokenizer.json ] || [ ! -f tools/tokenizer/tokenizer_config.json ]; }; then
  echo "Tokenizer: downloading ${BENCH_TOKENIZER_REPO} into tools/tokenizer/ (once)"
  if ! ./tools/download_tokenizer.sh "$BENCH_TOKENIZER_REPO"; then
    echo "WARNING: the tokenizer download failed (no Internet access, a gated model without HF_TOKEN, or a wrong name)."
    echo "Copy tokenizer.json and tokenizer_config.json of the model into tools/tokenizer/ (README, section 5.3)."
  fi
fi
unset HF_TOKEN
if [ "$(id -u)" = 0 ] && [ "$(stat -c %u . 2>/dev/null || echo 0)" != 0 ]; then
  echo "WARNING: running as root on files owned by user $(stat -c %u .): the results will belong to root."
  echo "Set BENCH_UID and BENCH_GID in .env (id -u, id -g): README, section 4.4."
fi
exec "$@"
