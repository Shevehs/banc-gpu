#!/usr/bin/env bash
# Télécharge le tokenizer du modèle mesuré dans outils/tokenizer/ (à faire une fois, sur une machine
# avec accès à Hugging Face), pour qu'il serve ensuite de tokenizer par défaut à balayage.sh,
# y compris sur des machines sans accès Internet.
# Usage : ./outils/telecharger_tokenizer.sh [dépôt Hugging Face]
# Exige l'environnement .venv-aiperf activé (la commande hf vient avec AIPerf).
set -euo pipefail
DEPOT="${1:-Qwen/Qwen3.8-Flash-Next-FP8}"
ICI="$(cd "$(dirname "$0")" && pwd)"
CIBLE="$ICI/tokenizer"
mkdir -p "$CIBLE"
hf download "$DEPOT" --local-dir "$CIBLE" \
  --include "tokenizer.json" --include "tokenizer_config.json" --include "vocab.json" \
  --include "merges.txt" --include "special_tokens_map.json" --include "LICENSE*"
rm -rf "$CIBLE/.cache"
echo "Tokenizer de $DEPOT dans $CIBLE :"
ls -la "$CIBLE"
[ -f "$CIBLE/tokenizer.json" ] || { echo "ATTENTION : tokenizer.json absent ; vérifie le nom du dépôt."; exit 1; }
echo "Vérifie la licence du modèle avant de versionner ces fichiers (git add outils/tokenizer)."
