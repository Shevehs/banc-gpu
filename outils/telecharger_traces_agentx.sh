#!/usr/bin/env bash
# Télécharge les traces AgentX (sessions d'agents de code publiées par SemiAnalysis, licence Apache-2.0 selon
# la fiche Hugging Face ; 393 traces, environ 1,85 Go) dans un dossier local, pour agentx.sh sans accès
# Internet (AGENTX_DOSSIER=<dossier>). À faire une fois, sur une machine avec accès à Hugging Face.
# Usage : ./outils/telecharger_traces_agentx.sh [dossier] [dépôt]
# Exige l'environnement .venv-aiperf activé (la commande hf vient avec AIPerf).
set -euo pipefail
CIBLE="${1:-traces-agentx}"
DEPOT="${2:-semianalysisai/cc-traces-weka-062126}"
hf download "$DEPOT" --repo-type dataset --local-dir "$CIBLE"
N=$(find "$CIBLE" -name '*.json' -not -path '*/.cache/*' | wc -l | tr -d ' ')
echo "$N fichier(s) JSON dans $CIBLE."
echo "agentx.sh lit les fichiers *.json d'un seul dossier : si les traces sont dans un sous-dossier, donne ce"
echo "sous-dossier dans AGENTX_DOSSIER. Vérifie la licence sur la fiche du jeu avant de le copier ailleurs."
