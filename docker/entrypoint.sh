#!/usr/bin/env bash
# Start of the bench container (README, section 5.2):
# - with BENCH_TOKENIZER set to a Hugging Face name (org/model) whose tokenizer is not yet in tools/tokenizer/ (its
#   .repo file names the model it was downloaded for), download it once (a failed download is not fatal: the message
#   says what to do, and the interface still starts);
# - warn if the container runs as root on files owned by someone else (Linux: results would belong to root);
# - then run the command (the interface), without HF_TOKEN in its environment.
set -u
NAME="${BENCH_TOKENIZER:-}"
if [[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$ ]] && [ ! -d "$NAME" ] \
   && { [ ! -f tools/tokenizer/tokenizer.json ] || [ ! -f tools/tokenizer/tokenizer_config.json ] \
        || [ "$(cat tools/tokenizer/.repo 2>/dev/null)" != "$NAME" ]; }; then
  echo "Tokenizer: downloading $NAME into tools/tokenizer/ (once)"
  if ! ./tools/download_tokenizer.sh "$NAME"; then
    echo "WARNING: the tokenizer download failed (no Internet access, a gated model without HF_TOKEN, or a wrong name)."
    echo "Copy tokenizer.json and tokenizer_config.json of the model into tools/tokenizer/ and leave BENCH_TOKENIZER empty (README, section 5.2)."
  fi
fi
unset HF_TOKEN
if [ "$(id -u)" = 0 ] && [ "$(stat -c %u . 2>/dev/null || echo 0)" != 0 ]; then
  echo "WARNING: running as root on files owned by user $(stat -c %u .): the results will belong to root."
  echo "Set BENCH_UID and BENCH_GID in .env (id -u, id -g): README, section 5.3."
fi
exec "$@"
