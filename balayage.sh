#!/usr/bin/env bash
# Balayage de concurrence avec AIPerf : un essai par niveau de concurrence.
# Usage : ./balayage.sh ETIQUETTE "1 2 4 8 16"
# Réglages par variables d'environnement (voir le README, § 3.1).
# URL vise vLLM directement ou une passerelle (LiteLLM). Les métriques de vLLM sont lues sur URL/metrics ;
# avec une passerelle ou plusieurs répliques, donner les URL /metrics de vLLM dans METRIQUES.
# Écrit pour bash 3.2 (macOS) et suivants ; testé avec bash 5.
set -euo pipefail
ETIQ="${1:?donner une étiquette, ex. essai-r1}"
if ! printf '%s' "$ETIQ" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$' || case "$ETIQ" in *..*) true ;; *) false ;; esac; then
  echo "Étiquette invalide : lettres, chiffres, point, tiret, souligné, en commençant par une lettre ou un chiffre ; ni .."; exit 1
fi
CONC="${2:-1 2 4 8 16}"
URL="${URL:-http://127.0.0.1:8000}"          # adresse du serveur vLLM
MODELE="${MODELE:-banc}"                     # nom donné par --served-model-name
ICI="$(cd "$(dirname "$0")" && pwd)"
TOKENIZER_DEFAUT=""; [ -f "$ICI/outils/tokenizer/tokenizer.json" ] && TOKENIZER_DEFAUT="$ICI/outils/tokenizer"
TOKENIZER="${TOKENIZER:-$TOKENIZER_DEFAUT}"  # tokenizer (chemin local ou nom Hugging Face) ; vide = nom du modèle
METRIQUES="${METRIQUES:-}"                   # URL /metrics de vLLM en plus de URL/metrics, séparées par des espaces
CLE_API="${CLE_API:-}"                       # clé d'API de la cible (AIPerf la masque dans ses journaux)
ISL="${ISL:-2048}"                           # jetons d'entrée par requête
OSL="${OSL:-256}"                            # jetons de sortie par requête (imposés : min_tokens + ignore_eos)
DUREE="${DUREE:-}"                           # secondes par palier ; vide = nombre de requêtes
REQ_PAR_CONC="${REQ_PAR_CONC:-8}"            # sans DUREE : requêtes par palier = 8 x concurrence, au moins 16
TEMPERATURE="${TEMPERATURE:-0.7}"            # fixée pour que l'acceptation MTP soit comparable d'un essai à l'autre
RACINE="${RACINE:-resultats}"                # dossier des résultats
ECRASER="${ECRASER:-}"                       # 1 = remplacer des résultats existants de même étiquette
EXTRA="${EXTRA:-}"                           # options AIPerf supplémentaires, ex. "--use-server-token-count --api-key XXX"
# Graine différente pour chaque étiquette (donc pour chaque répétition), sauf si SEED est fixée.
SEED="${SEED:-$(printf '%s' "$ETIQ" | cksum | cut -d' ' -f1)}"
if [ -d "$RACINE/$ETIQ" ] && [ -n "$(ls -A "$RACINE/$ETIQ" 2>/dev/null)" ]; then
  if [ "$ECRASER" = "1" ]; then rm -rf "${RACINE:?}/$ETIQ"
  else echo "$RACINE/$ETIQ existe déjà : change d'étiquette, ou relance avec ECRASER=1 pour le remplacer."; exit 1; fi
fi
mkdir -p "$RACINE/$ETIQ"
# Instantané de la configuration du serveur (version, cache, parallélisme par l'empreinte des réponses).
CLE_API="$CLE_API" python3 "$ICI/outils/config_serveur.py" --url "$URL" --modele "$MODELE" --metriques "$METRIQUES" \
  --sortie "$RACINE/$ETIQ/config_serveur.json" || echo "   (configuration du serveur non relevée)"
# Tokenizer local : AIPerf 0.12 le cherche dans le cache Hugging Face si HF_HUB_OFFLINE est actif, et échoue.
# On retire donc ces variables pour AIPerf seulement ; un dossier local ne déclenche aucun téléchargement.
SANS_HORS_LIGNE=()
[ -n "$TOKENIZER" ] && [ -d "$TOKENIZER" ] && SANS_HORS_LIGNE=(env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE)
for C in $CONC; do
  DIR="$RACINE/$ETIQ/c$C"
  ARGS=(-m "$MODELE" --endpoint-type chat --streaming -u "$URL"
        --isl "$ISL" --osl "$OSL" --extra-inputs "min_tokens:$OSL" --extra-inputs ignore_eos:true
        --extra-inputs "temperature:$TEMPERATURE"
        --concurrency "$C" --warmup-request-count "$C" --random-seed "$SEED" --artifact-dir "$DIR")
  if [ -n "$DUREE" ]; then
    ARGS+=(--benchmark-duration "$DUREE")
  else
    N=$(( C * REQ_PAR_CONC )); [ "$N" -lt 16 ] && N=16
    ARGS+=(--request-count "$N")
  fi
  [ -n "$TOKENIZER" ] && ARGS+=(--tokenizer "$TOKENIZER")
  [ -n "$CLE_API" ] && ARGS+=(--api-key "$CLE_API")
  # shellcheck disable=SC2206
  [ -n "$METRIQUES" ] && ARGS+=(--server-metrics $METRIQUES)
  # shellcheck disable=SC2206
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  echo "== $ETIQ : concurrence $C =="
  T0=$(date +%s)
  if ! ${SANS_HORS_LIGNE[@]+"${SANS_HORS_LIGNE[@]}"} aiperf profile "${ARGS[@]}" > "$DIR.log" 2>&1; then
    echo "ÉCHEC à la concurrence $C : lire $DIR.log"; exit 1
  fi
  echo "   $(( $(date +%s) - T0 )) s"
done
echo "Terminé. Résultats dans $RACINE/$ETIQ/ (AIPerf ne s'arrête pas sur les requêtes en erreur : vérifie la colonne err du dépouillement)."
