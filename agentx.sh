#!/usr/bin/env bash
# AgentX stress test: replays the coding-agent sessions published by SemiAnalysis (InferenceX AgentX)
# with AIPerf's locked scenario (--scenario inferencex-agentx-mvp), one trial per concurrency level
# (number of sessions replayed at the same time). Analysis: python -m report aiperf results-agentx ...
# Usage: ./agentx.sh LABEL "1 2 4 8"
# Settings via environment variables (README, section 6.3). Same connection fields as sweep.sh.
# Written for bash 3.2 and later; tested with bash 5.
set -euo pipefail
LABEL="${1:?give a label, e.g. agentx-r1}"
if ! printf '%s' "$LABEL" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$' || case "$LABEL" in *..*) true ;; *) false ;; esac; then
  echo "Invalid label: letters, digits, dot, hyphen, underscore, starting with a letter or a digit; no .."; exit 1
fi
CONCURRENCIES="${2:-1 2 4 8}"
URL="${BENCH_URL:-http://127.0.0.1:8000}"
MODEL="${BENCH_MODEL:-bench}"
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/tools/settings.py" || true   # warns about variables from older versions that are no longer read
DEFAULT_TOKENIZER=""; [ -f "$HERE/tools/tokenizer/tokenizer.json" ] && DEFAULT_TOKENIZER="$HERE/tools/tokenizer"
TOKENIZER="${BENCH_TOKENIZER:-$DEFAULT_TOKENIZER}"
METRICS="${BENCH_METRICS:-}"
API_KEY="${BENCH_API_KEY:-}"
DURATION="${BENCH_DURATION:-1800}"                       # seconds per level; the scenario requires at least 900
ROOT_DIR="${BENCH_ROOT_DIR:-results-agentx}"
OVERWRITE="${BENCH_OVERWRITE:-}"
# Traces: by default, the dataset pinned by the scenario, downloaded from Hugging Face:
#   BENCH_AGENTX_VARIANT empty -> semianalysisai/cc-traces-weka-062126 (full contexts)
#   BENCH_AGENTX_VARIANT=256k  -> semianalysisai/cc-traces-weka-062126-256k (requests over 256,000 tokens removed
#                            by SemiAnalysis; for a --max-model-len of about 256,000; also accepted by the scenario)
# BENCH_AGENTX_DIR = local traces folder (JSON files in Weka format): AIPerf then cannot verify the
# dataset, and the trial is marked submission_valid=false (--unsafe-override).
AGENTX_VARIANT="${BENCH_AGENTX_VARIANT:-}"
AGENTX_DIR="${BENCH_AGENTX_DIR:-}"
SHORT_RUN="${BENCH_SHORT_RUN:-}"               # 1 = levels shorter than 900 s (check only, not comparable)
# Maximum context (tokens): AIPerf discards traces whose peak context exceeds this value
# (--max-context-length). For a short-context model (e.g. a small server, 32,768): only the sessions that fit
# are replayed, a subset biased toward short sessions, hence not comparable to published results.
MAX_CONTEXT="${BENCH_MAX_CONTEXT:-}"
SEED="${BENCH_SEED:-$(printf '%s' "$LABEL" | cksum | cut -d' ' -f1)}"
# All checks before creating the results folder.
case "$DURATION" in ""|*[!0-9]*) echo "BENCH_DURATION=$DURATION: integer number of seconds expected."; exit 1 ;; esac
case "$MAX_CONTEXT" in *[!0-9]*|0*) echo "BENCH_MAX_CONTEXT: integer number of tokens (> 0) expected."; exit 1 ;; esac
case "$AGENTX_VARIANT" in ""|256k) ;; *) echo "BENCH_AGENTX_VARIANT: empty or 256k."; exit 1 ;; esac
if [ -n "$AGENTX_DIR" ] && [ ! -d "$AGENTX_DIR" ]; then echo "BENCH_AGENTX_DIR=$AGENTX_DIR: folder not found."; exit 1; fi
for C in $CONCURRENCIES; do case "$C" in *[!0-9]*) echo "Invalid concurrency: $C"; exit 1 ;; esac; done
if [ "$SHORT_RUN" != "1" ] && [ "$DURATION" -lt 900 ]; then
  echo "BENCH_DURATION=$DURATION: the AgentX scenario requires at least 900 s per level (BENCH_SHORT_RUN=1 for a simple check)."; exit 1
fi
if [ -d "$ROOT_DIR/$LABEL" ] && [ -n "$(ls -A "$ROOT_DIR/$LABEL" 2>/dev/null)" ]; then
  if [ "$OVERWRITE" = "1" ]; then rm -rf "${ROOT_DIR:?}/$LABEL"
  else echo "$ROOT_DIR/$LABEL already exists: change the label, or rerun with BENCH_OVERWRITE=1 to replace it."; exit 1; fi
fi
mkdir -p "$ROOT_DIR/$LABEL"
BENCH_API_KEY="$API_KEY" python3 "$HERE/tools/server_config.py" --url "$URL" --model "$MODEL" --metrics "$METRICS" \
  --output "$ROOT_DIR/$LABEL/server_config.json" || echo "   (server configuration not collected)"
SOURCE=(--hf-weka-dataset semianalysisai/cc-traces-weka-062126); DATASET=semianalysisai/cc-traces-weka-062126; OVERRIDE=()
if [ "$AGENTX_VARIANT" = "256k" ]; then
  SOURCE=(--public-dataset semianalysis_cc_traces_weka_062126_256k); DATASET=semianalysisai/cc-traces-weka-062126-256k
fi
if [ -n "$AGENTX_DIR" ]; then
  DATASET="local folder"
  SOURCE=(--custom-dataset-type weka-trace --input-file "$AGENTX_DIR"); OVERRIDE=(--unsafe-override)
  echo "   Local traces: trial marked submission_valid=false (dataset not verified by AIPerf)."
  [ -n "$AGENTX_VARIANT" ] && echo "   BENCH_AGENTX_VARIANT=$AGENTX_VARIANT ignored: the folder determines the dataset."
fi
[ "$SHORT_RUN" = "1" ] && OVERRIDE=(--unsafe-override) && echo "   BENCH_SHORT_RUN=1: check only, results not comparable."
[ -n "$MAX_CONTEXT" ] && echo "   BENCH_MAX_CONTEXT=$MAX_CONTEXT: only the traces that fit are replayed (subset, not comparable)."
VALID=yes; { [ -n "$AGENTX_DIR" ] || [ "$SHORT_RUN" = "1" ]; } && VALID=no
printf 'dataset=%s\ndirectory=%s\nduration=%s\nshort_run=%s\nmax_context=%s\nsubmission_valid=%s\n' "$DATASET" "$AGENTX_DIR" "$DURATION" "$SHORT_RUN" "$MAX_CONTEXT" "$VALID" > "$ROOT_DIR/$LABEL/agentx.txt"
# Local tokenizer: AIPerf 0.12 looks for it in the Hugging Face cache if HF_HUB_OFFLINE is set, and fails
# (same workaround as sweep.sh). The online traces can still be downloaded.
OFFLINE_UNSET=()
[ -n "$TOKENIZER" ] && [ -d "$TOKENIZER" ] && OFFLINE_UNSET=(env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE)
# Loading the 393 traces (download included on the first run) takes several minutes, more than the 300 s
# AIPerf allows by default (AIPERF_DATASET_CONFIGURATION_TIMEOUT; the profile configuration timeout,
# AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT, must be at least equal). Values set in the terminal are respected.
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="$(awk -v a="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-0}" \
  -v b="$AIPERF_DATASET_CONFIGURATION_TIMEOUT" 'BEGIN { print (a + 0 > b + 0) ? a : b }')"   # AIPerf requires >= the first one
for C in $CONCURRENCIES; do
  DIR="$ROOT_DIR/$LABEL/c$C"
  ARGS=(-m "$MODEL" --endpoint-type chat --streaming -u "$URL" --scenario inferencex-agentx-mvp
        "${SOURCE[@]}" --concurrency "$C" --benchmark-duration "$DURATION" --random-seed "$SEED" --artifact-dir "$DIR")
  [ -n "$MAX_CONTEXT" ] && ARGS+=(--max-context-length "$MAX_CONTEXT")
  [ -n "$TOKENIZER" ] && ARGS+=(--tokenizer "$TOKENIZER")
  [ -n "$API_KEY" ] && ARGS+=(--api-key "$API_KEY")
  # shellcheck disable=SC2206
  [ -n "$METRICS" ] && ARGS+=(--server-metrics $METRICS)
  echo "== $LABEL: $C concurrent session(s), $DURATION s =="
  T0=$(date +%s)
  if ! ${OFFLINE_UNSET[@]+"${OFFLINE_UNSET[@]}"} aiperf profile "${ARGS[@]}" ${OVERRIDE[@]+"${OVERRIDE[@]}"} > "$DIR.log" 2>&1; then
    echo "FAILED at concurrency $C: see $DIR.log"
    grep -q -E "All traces rejected|No eligible traces" "$DIR.log" 2>/dev/null && echo "   No trace fits within BENCH_MAX_CONTEXT=$MAX_CONTEXT: increase it (and vLLM's --max-model-len)."
    exit 1
  fi
  echo "   $(( $(date +%s) - T0 )) s"
  SEL=$(grep -o -E "scanned [0-9,]+, rejected_by_maxctx [0-9,]+( \([^)]*\))?, eligible [0-9]+(,[0-9]{3})*" "$DIR.log" 2>/dev/null | head -1 || true)
  [ -n "$SEL" ] && echo "   traces: $SEL"
done
echo "Done. Results in $ROOT_DIR/$LABEL/; analysis: python -m report aiperf $ROOT_DIR --target 30 (GPUs deduced from the fingerprint; --gpu N to force)"
