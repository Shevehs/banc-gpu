#!/usr/bin/env bash
# Stress test AgentX : rejoue les sessions d'agents de code publiées par SemiAnalysis (InferenceX AgentX)
# avec le scénario verrouillé d'AIPerf (--scenario inferencex-agentx-mvp), un essai par niveau de concurrence
# (nombre de sessions rejouées en même temps). Dépouillement : python depouiller.py resultats-agentx ...
# Usage : ./agentx.sh ETIQUETTE "1 2 4 8"
# Réglages par variables d'environnement (README, § 4). Mêmes champs de connexion que balayage.sh.
# Écrit pour bash 3.2 (macOS) et suivants ; testé avec bash 5.
set -euo pipefail
ETIQ="${1:?donner une étiquette, ex. agentx-r1}"
if ! printf '%s' "$ETIQ" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$' || case "$ETIQ" in *..*) true ;; *) false ;; esac; then
  echo "Étiquette invalide : lettres, chiffres, point, tiret, souligné, en commençant par une lettre ou un chiffre ; ni .."; exit 1
fi
CONC="${2:-1 2 4 8}"
URL="${URL:-http://127.0.0.1:8000}"
MODELE="${MODELE:-banc}"
ICI="$(cd "$(dirname "$0")" && pwd)"
TOKENIZER_DEFAUT=""; [ -f "$ICI/outils/tokenizer/tokenizer.json" ] && TOKENIZER_DEFAUT="$ICI/outils/tokenizer"
TOKENIZER="${TOKENIZER:-$TOKENIZER_DEFAUT}"
METRIQUES="${METRIQUES:-}"
CLE_API="${CLE_API:-}"
DUREE="${DUREE:-1800}"                       # secondes par palier ; le scénario exige au moins 900
RACINE="${RACINE:-resultats-agentx}"
ECRASER="${ECRASER:-}"
# Traces : par défaut, le jeu épinglé par le scénario, téléchargé depuis Hugging Face :
#   AGENTX_VARIANTE vide  -> semianalysisai/cc-traces-weka-062126 (contextes complets)
#   AGENTX_VARIANTE=256k  -> semianalysisai/cc-traces-weka-062126-256k (requêtes de plus de 256 000 jetons retirées
#                            par SemiAnalysis ; pour un --max-model-len d'environ 256 000 ; aussi accepté par le scénario)
# AGENTX_DOSSIER = dossier local de traces (fichiers JSON au format Weka) : AIPerf ne peut alors pas vérifier le
# jeu, l'essai est marqué submission_valid=false (--unsafe-override).
AGENTX_VARIANTE="${AGENTX_VARIANTE:-}"
AGENTX_DOSSIER="${AGENTX_DOSSIER:-}"
ESSAI_COURT="${ESSAI_COURT:-}"               # 1 = paliers de moins de 900 s (vérification seulement, non comparable)
# Contexte maximal (jetons) : AIPerf écarte les traces dont le contexte de pointe dépasse cette valeur
# (--max-context-length). Pour un modèle à contexte court (ex. Mac, 32 768) : seules les sessions qui tiennent
# sont rejouées, sous-ensemble biaisé vers les sessions courtes, donc non comparable aux résultats publiés.
CONTEXTE_MAX="${CONTEXTE_MAX:-}"
SEED="${SEED:-$(printf '%s' "$ETIQ" | cksum | cut -d' ' -f1)}"
# Tous les contrôles avant de créer le dossier de résultats.
case "$DUREE" in ""|*[!0-9]*) echo "DUREE=$DUREE : nombre entier de secondes attendu."; exit 1 ;; esac
case "$CONTEXTE_MAX" in *[!0-9]*|0*) echo "CONTEXTE_MAX : nombre entier de jetons (> 0) attendu."; exit 1 ;; esac
case "$AGENTX_VARIANTE" in ""|256k) ;; *) echo "AGENTX_VARIANTE : vide ou 256k."; exit 1 ;; esac
if [ -n "$AGENTX_DOSSIER" ] && [ ! -d "$AGENTX_DOSSIER" ]; then echo "AGENTX_DOSSIER=$AGENTX_DOSSIER : dossier introuvable."; exit 1; fi
for C in $CONC; do case "$C" in *[!0-9]*) echo "Concurrence invalide : $C"; exit 1 ;; esac; done
if [ "$ESSAI_COURT" != "1" ] && [ "$DUREE" -lt 900 ]; then
  echo "DUREE=$DUREE : le scénario AgentX exige au moins 900 s par palier (ESSAI_COURT=1 pour une simple vérification)."; exit 1
fi
if [ -d "$RACINE/$ETIQ" ] && [ -n "$(ls -A "$RACINE/$ETIQ" 2>/dev/null)" ]; then
  if [ "$ECRASER" = "1" ]; then rm -rf "${RACINE:?}/$ETIQ"
  else echo "$RACINE/$ETIQ existe déjà : change d'étiquette, ou relance avec ECRASER=1 pour le remplacer."; exit 1; fi
fi
mkdir -p "$RACINE/$ETIQ"
CLE_API="$CLE_API" python3 "$ICI/outils/config_serveur.py" --url "$URL" --modele "$MODELE" --metriques "$METRIQUES" \
  --sortie "$RACINE/$ETIQ/config_serveur.json" || echo "   (configuration du serveur non relevée)"
SOURCE=(--hf-weka-dataset semianalysisai/cc-traces-weka-062126); JEU=semianalysisai/cc-traces-weka-062126; OVERRIDE=()
if [ "$AGENTX_VARIANTE" = "256k" ]; then
  SOURCE=(--public-dataset semianalysis_cc_traces_weka_062126_256k); JEU=semianalysisai/cc-traces-weka-062126-256k
fi
if [ -n "$AGENTX_DOSSIER" ]; then
  JEU="dossier local"
  SOURCE=(--custom-dataset-type weka-trace --input-file "$AGENTX_DOSSIER"); OVERRIDE=(--unsafe-override)
  echo "   Traces locales : essai marqué submission_valid=false (jeu non vérifié par AIPerf)."
  [ -n "$AGENTX_VARIANTE" ] && echo "   AGENTX_VARIANTE=$AGENTX_VARIANTE ignorée : c'est le dossier qui choisit le jeu."
fi
[ "$ESSAI_COURT" = "1" ] && OVERRIDE=(--unsafe-override) && echo "   ESSAI_COURT=1 : vérification seulement, résultats non comparables."
[ -n "$CONTEXTE_MAX" ] && echo "   CONTEXTE_MAX=$CONTEXTE_MAX : seules les traces qui tiennent sont rejouées (sous-ensemble, non comparable)."
VALIDE=oui; { [ -n "$AGENTX_DOSSIER" ] || [ "$ESSAI_COURT" = "1" ]; } && VALIDE=non
printf 'jeu=%s\ndossier=%s\nduree=%s\nessai_court=%s\ncontexte_max=%s\nsubmission_valid=%s\n' "$JEU" "$AGENTX_DOSSIER" "$DUREE" "$ESSAI_COURT" "$CONTEXTE_MAX" "$VALIDE" > "$RACINE/$ETIQ/agentx.txt"
# Tokenizer local : AIPerf 0.12 le cherche dans le cache Hugging Face si HF_HUB_OFFLINE est actif, et échoue
# (même contournement que balayage.sh). Les traces en ligne restent téléchargeables.
SANS_HORS_LIGNE=()
[ -n "$TOKENIZER" ] && [ -d "$TOKENIZER" ] && SANS_HORS_LIGNE=(env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE)
for C in $CONC; do
  DIR="$RACINE/$ETIQ/c$C"
  ARGS=(-m "$MODELE" --endpoint-type chat --streaming -u "$URL" --scenario inferencex-agentx-mvp
        "${SOURCE[@]}" --concurrency "$C" --benchmark-duration "$DUREE" --random-seed "$SEED" --artifact-dir "$DIR")
  [ -n "$CONTEXTE_MAX" ] && ARGS+=(--max-context-length "$CONTEXTE_MAX")
  [ -n "$TOKENIZER" ] && ARGS+=(--tokenizer "$TOKENIZER")
  [ -n "$CLE_API" ] && ARGS+=(--api-key "$CLE_API")
  # shellcheck disable=SC2206
  [ -n "$METRIQUES" ] && ARGS+=(--server-metrics $METRIQUES)
  echo "== $ETIQ : $C session(s) simultanée(s), $DUREE s =="
  T0=$(date +%s)
  if ! ${SANS_HORS_LIGNE[@]+"${SANS_HORS_LIGNE[@]}"} aiperf profile "${ARGS[@]}" ${OVERRIDE[@]+"${OVERRIDE[@]}"} > "$DIR.log" 2>&1; then
    echo "ÉCHEC à la concurrence $C : lire $DIR.log"
    grep -q -E "All traces rejected|No eligible traces" "$DIR.log" 2>/dev/null && echo "   Aucune trace ne tient dans CONTEXTE_MAX=$CONTEXTE_MAX : augmente-le (et --max-model-len de vLLM)."
    exit 1
  fi
  echo "   $(( $(date +%s) - T0 )) s"
  SEL=$(grep -o -E "scanned [0-9,]+, rejected_by_maxctx [0-9,]+( \([^)]*\))?, eligible [0-9]+(,[0-9]{3})*" "$DIR.log" 2>/dev/null | head -1 || true)
  [ -n "$SEL" ] && echo "   traces : $SEL"
done
echo "Terminé. Résultats dans $RACINE/$ETIQ/ ; dépouillement : python depouiller.py $RACINE --cible 30 --gpu <GPU>"
