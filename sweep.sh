#!/usr/bin/env bash
# Concurrency sweep with AIPerf: one trial per concurrency level.
# Usage: ./sweep.sh LABEL "1 2 4 8 16"
# Settings via environment variables (see the README, section 7.1).
# URL points at vLLM directly or at a gateway (LiteLLM). The vLLM metrics are read from URL/metrics;
# with a gateway or several replicas, give the vLLM /metrics URLs in BENCH_METRICS.
# Written for bash 3.2 and later; tested with bash 5.
set -euo pipefail
LABEL="${1:?give a label, e.g. trial-r1}"
if ! printf '%s' "$LABEL" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$' || case "$LABEL" in *..*) true ;; *) false ;; esac; then
  echo "Invalid label: letters, digits, dot, hyphen, underscore, starting with a letter or a digit; no .."; exit 1
fi
CONCURRENCIES="${2:-1 2 4 8 16}"
URL="${BENCH_URL:-http://127.0.0.1:8000}"          # address of the vLLM server
MODEL="${BENCH_MODEL:-bench}"                     # name given by --served-model-name
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/tools/settings.py" || true   # warns about variables from older versions that are no longer read
LOCAL_TOKENIZER="$HERE/tools/tokenizer"
TOKENIZER="${BENCH_TOKENIZER:-}"      # Hugging Face name or folder; empty = tools/tokenizer if it holds one
# tools/tokenizer is used for an empty value, and for the name it was downloaded for (its .repo file)
if [ -f "$LOCAL_TOKENIZER/tokenizer.json" ] && { [ -z "$TOKENIZER" ] || [ "$(cat "$LOCAL_TOKENIZER/.repo" 2>/dev/null)" = "$TOKENIZER" ]; }; then TOKENIZER="$LOCAL_TOKENIZER"; fi
METRICS="${BENCH_METRICS:-}"                   # vLLM /metrics URLs in addition to URL/metrics, separated by spaces
API_KEY="${BENCH_API_KEY:-}"                       # API key for the target (AIPerf masks it in its logs)
ISL="${BENCH_ISL:-2048}"                           # input tokens per request
OSL="${BENCH_OSL:-256}"                            # output tokens per request (forced: min_tokens + ignore_eos)
DURATION="${BENCH_DURATION:-}"                           # seconds per level; empty = request count
REQUESTS_PER_CONCURRENCY="${BENCH_REQUESTS_PER_CONCURRENCY:-8}"            # without DURATION: requests per level = 8 x concurrency, at least 16
TEMPERATURE="${BENCH_TEMPERATURE:-0.7}"            # fixed so that MTP acceptance is comparable from one trial to another
ROOT_DIR="${BENCH_ROOT_DIR:-results}"                # results directory
OVERWRITE="${BENCH_OVERWRITE:-}"                       # 1 = replace existing results with the same label
EXTRA="${BENCH_EXTRA:-}"                           # additional AIPerf options, e.g. "--use-server-token-count --api-key XXX"
# A different seed for each label (hence for each repetition), unless BENCH_SEED is set.
SEED="${BENCH_SEED:-$(printf '%s' "$LABEL" | cksum | cut -d' ' -f1)}"
# This bench needs AIPerf 0.13 or later: stop before anything is replaced or created.
AIPERF_VERSION="$(aiperf --version 2>/dev/null | tail -n 1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n 1 || true)"
case "$AIPERF_VERSION" in
  0.[0-9]|0.[0-9].*|0.1[0-2]|0.1[0-2].*)
    echo "AIPerf $AIPERF_VERSION is installed: this bench needs AIPerf 0.13.0 or later (pip install aiperf==0.13.0)."; exit 1 ;;
esac
# A model name the server does not know makes every request fail at once: say so before anything is replaced or created.
CHECK=0
BENCH_API_KEY="$API_KEY" python3 "$HERE/tools/server_config.py" --url "$URL" --model "$MODEL" --check-model || CHECK=$?
if [ "$CHECK" = "3" ]; then
  echo "Stopped before any measurement: set BENCH_MODEL to a model the server serves (it is \"bench\" when BENCH_MODEL is not set)."; exit 1
fi
if [ -d "$ROOT_DIR/$LABEL" ] && [ -n "$(ls -A "$ROOT_DIR/$LABEL" 2>/dev/null)" ]; then
  if [ "$OVERWRITE" = "1" ]; then rm -rf "${ROOT_DIR:?}/$LABEL"
  else echo "$ROOT_DIR/$LABEL already exists: change the label, or rerun with BENCH_OVERWRITE=1 to replace it."; exit 1; fi
fi
mkdir -p "$ROOT_DIR/$LABEL"
# Snapshot of the server configuration (version, cache, parallelism from the response fingerprint).
BENCH_API_KEY="$API_KEY" python3 "$HERE/tools/server_config.py" --url "$URL" --model "$MODEL" --metrics "$METRICS" \
  --output "$ROOT_DIR/$LABEL/server_config.json" || echo "   (server configuration not collected)"
# Local tokenizer: AIPerf looks for it in the Hugging Face cache if HF_HUB_OFFLINE is set, and fails.
# So these variables are unset for AIPerf only; a local folder triggers no download.
OFFLINE_UNSET=()
[ -n "$TOKENIZER" ] && [ -d "$TOKENIZER" ] && OFFLINE_UNSET=(env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE)
for C in $CONCURRENCIES; do
  DIR="$ROOT_DIR/$LABEL/c$C"
  ARGS=(-m "$MODEL" --endpoint-type chat --streaming -u "$URL"
        --isl "$ISL" --osl "$OSL" --extra-inputs "min_tokens:$OSL" --extra-inputs ignore_eos:true
        --extra-inputs "temperature:$TEMPERATURE"
        --concurrency "$C" --warmup-request-count "$C" --random-seed "$SEED" --artifact-dir "$DIR")
  if [ -n "$DURATION" ]; then
    ARGS+=(--benchmark-duration "$DURATION")
  else
    N=$(( C * REQUESTS_PER_CONCURRENCY )); [ "$N" -lt 16 ] && N=16
    ARGS+=(--request-count "$N")
  fi
  [ -n "$TOKENIZER" ] && ARGS+=(--tokenizer "$TOKENIZER")
  [ -n "$API_KEY" ] && ARGS+=(--api-key "$API_KEY")
  # shellcheck disable=SC2206
  [ -n "$METRICS" ] && ARGS+=(--server-metrics $METRICS)
  # shellcheck disable=SC2206
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  echo "== $LABEL: concurrency $C =="
  T0=$(date +%s)
  if ! ${OFFLINE_UNSET[@]+"${OFFLINE_UNSET[@]}"} aiperf profile "${ARGS[@]}" > "$DIR.log" 2>&1; then
    echo "FAILED at concurrency $C: see $DIR.log"; exit 1
  fi
  echo "   $(( $(date +%s) - T0 )) s"
done
echo "Done. Results in $ROOT_DIR/$LABEL/ (AIPerf does not stop on failed requests: check the err column of the analysis)."
