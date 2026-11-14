# banc-gpu

Banc de mesure de capacité pour un modèle servi par vLLM : pour une installation donnée, combien de
requêtes simultanées (ou de tâches d'agents) elle tient, et à quel débit par GPU, tant que chaque
requête reçoit ses jetons à une vitesse visée (l'**interactivité**, en jetons/s par requête).

Deux mesures complémentaires :

| Mesure | Outil | Cache de préfixes de vLLM | Ce qu'elle donne |
|---|---|---|---|
| **Charge synthétique** | AIPerf (`balayage.sh`) | **désactivé** | débit de référence, prudent, à longueurs d'entrée et de sortie fixées |
| **Agents de test** | `agents_test/` | **activé** | débit sous une charge d'agents (contexte qui grossit, appels d'outils, sous-agents) |

Écrit d'après le code de **vLLM au commit [`7f855f6`](https://github.com/vllm-project/vllm/tree/7f855f605892656ebf7265642e5d22a34a6ae731)** et testé avec **AIPerf 0.12.0** contre le serveur
factice (§ 10), **pas encore exécuté contre un vrai vLLM** : contrôle les sorties réelles au § 2.4 (sources en fin de document).

Les agents de test sont trois profils (`revue_code` avec sous-agents, `pentest` pas à pas, `rapport`).
Leurs outils sont simulés : ils renvoient du texte fabriqué et n'agissent sur aucun système.

---

## 1. Requis

### 1.1 Ce qui doit déjà être en place

| Requis | Détail | Vérification |
|---|---|---|
| **vLLM installé et le modèle servi** | API compatible OpenAI sur `http://<hôte>:<port>`. Le banc ne démarre pas vLLM. | `curl -s http://<hôte>:8000/v1/models` liste le modèle |
| **`/metrics` de vLLM joignable** depuis la machine cliente | Actif par défaut ; il donne préemptions, cache, décodage spéculatif, occupation du cache KV. Il n'exige pas la clé d'API (§ 1.2). | `curl -s http://<hôte>:8000/metrics \| grep -c '^vllm:'` > 0 |
| **LiteLLM installé** (optionnel) | Seulement pour mesurer à travers la passerelle. Modèle déclaré en `hosted_vllm/<nom>` avec `api_base: http://<hôte>:8000/v1` et, si vLLM a une clé, `api_key` ([LiteLLM, vLLM](https://docs.litellm.ai/docs/providers/vllm) ; [configuration](https://docs.litellm.ai/docs/proxy/configs)). | `curl -s -H "Authorization: Bearer <clé>" http://<vm>:4000/v1/models` |
| **Python 3.11 à 3.13** sur la machine cliente | Exigé par AIPerf 0.12. | `python3 --version` |
| **bash** | Testé avec bash 5 ; écrit pour bash 3.2 (macOS), vérifié seulement avec un bash 3.2 compilé sous Linux. | `bash --version` |
| **Réseau** | La machine cliente atteint vLLM (et LiteLLM). Aucun accès Internet n'est nécessaire une fois le tokenizer copié dans `outils/tokenizer/` (§ 2.3). | — |

La **machine cliente** est celle qui lance les mesures : le même poste que vLLM pour un essai local, ou
une VM sur le même réseau privé que les serveurs GPU. Ne lance pas les clients sur les serveurs mesurés.

### 1.2 Options de lancement de vLLM selon la mesure

Le banc lit la configuration du serveur, mais c'est toi qui choisis comment vLLM est lancé. Options qui
comptent pour le banc :

| Option de `vllm serve` | Charge synthétique | Agents de test | Pourquoi |
|---|---|---|---|
| `--served-model-name <nom>` | conseillé | conseillé | nom court à donner au banc (`MODELE`) |
| `--no-enable-prefix-caching` | **requis** | **ne pas mettre** | AIPerf réutilise ses invites : avec le cache, le débit serait gonflé. Les agents, eux, relisent leur contexte : le cache fait partie de ce qu'on mesure. |
| `--max-model-len` | ≥ entrée + sortie (ex. 2 048 + 256) | ≥ ≈ 135 000 × `--echelle`, plus la sortie | les agents atteignent ≈ 135 000 jetons de contexte à l'échelle 1 (estimé avec le vocabulaire Qwen) |
| `--max-num-seqs` | ≥ la plus haute concurrence testée | ≥ le plus haut palier (× 3 pour `revue_code`, qui lance des sous-agents en parallèle) | sinon les requêtes en trop attendent |
| `--enable-prompt-tokens-details` | — | conseillé | chaque réponse indique les jetons lus en cache |
| `--enable-auto-tool-choice --tool-call-parser <parseur>` | — | requis pour le mode `autonome` seulement | appel d'outils choisi par le modèle |
| `--api-key <clé>` | optionnel | optionnel | à donner au banc (`CLE_API`) ; voir la mise en garde ci-dessous |
| `--fingerprint-mode` | laisser par défaut (`full`) | idem | le banc lit le parallélisme (TP, PP, DP, EP) dans l'empreinte des réponses ; en mode `hash`, il est masqué |

**`--api-key` ne suffit pas à protéger vLLM.** Elle ne s'applique qu'aux chemins `/v1`, `/v2`, `/inference`
et `/cohere` ; les autres restent ouverts, dont `/metrics` et `/invocations`, qui donne la même inférence
que `/v1` ([vLLM, `cli_args.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/common/cli_args.py#L337-L347) ;
[sécurité vLLM](https://docs.vllm.ai/en/latest/usage/security.html#api-key-authentication-limitations)).
Ouvre le port de vLLM au seul client de test (pare-feu, réseau privé). Côté serveur, `DO_NOT_TRACK=1`
coupe l'envoi de statistiques d'usage de vLLM ([`envs.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/envs.py), [`usage_lib.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/usage/usage_lib.py)).

Changer le cache de préfixes impose de **redémarrer vLLM** entre la charge synthétique et les agents.

### 1.3 Ce que le banc installe

```bash
git clone https://github.com/Shevehs/banc-gpu.git ~/banc && cd ~/banc
python3 -m venv .venv-aiperf && source .venv-aiperf/bin/activate
pip install aiperf==0.12.0 "openai>=1.40"
aiperf --version && python -c "import openai, aiohttp; print('ok')"
```

AIPerf installe aussi `aiohttp` (interface) et la commande `hf` (tokenizer). Sans accès Internet,
`pip` doit pointer vers un miroir de paquets.

---

## 2. Préparer la machine cliente (une fois)

### 2.1 Activer l'environnement (dans chaque nouveau terminal)

```bash
cd ~/banc && source .venv-aiperf/bin/activate
```

### 2.2 Renseigner la connexion

```bash
cp outils/cible.env.exemple cible.env     # cible.env est ignoré par git : il contient la clé
```

Remplis `cible.env`, puis charge-le dans chaque nouveau terminal : `source cible.env`.

| Variable | Sens | Exemple |
|---|---|---|
| `URL` | **cible** des requêtes : vLLM directement (mesure des GPU) ou LiteLLM (mesure à travers la passerelle) | `http://10.0.0.5:8000` |
| `MODELE` | nom servi par vLLM (`--served-model-name`) ou alias LiteLLM | `banc` |
| `METRIQUES` | URL `/metrics` **de vLLM**, séparées par des espaces, une par réplique. Vide si `URL` vise déjà vLLM ; **obligatoire** si `URL` vise LiteLLM | `http://10.0.0.5:8000/metrics` |
| `CLE_API` | clé de la cible (`--api-key` de vLLM, ou clé LiteLLM). Envoyée à la cible seulement, jamais journalisée | — |
| `TOKENIZER` | vide = `outils/tokenizer/` (§ 2.3) ; sinon chemin local (dossier) ou nom Hugging Face. Les agents ne lisent qu'un dossier local | vide |

Ne mets pas `HF_HUB_OFFLINE=1` dans ce fichier : avec un tokenizer local, AIPerf 0.12 échoue quand cette
variable est active (`balayage.sh` la retire de toute façon pour AIPerf).

### 2.3 Copier le tokenizer du modèle mesuré

AIPerf en a besoin pour fabriquer des invites de la bonne longueur ; les agents s'en servent pour ajuster
leurs tailles. Sur une machine avec accès à Hugging Face :

```bash
./outils/telecharger_tokenizer.sh <dépôt Hugging Face du modèle>   # dépose les fichiers dans outils/tokenizer/
ls outils/tokenizer/tokenizer.json
```

Sans accès à Hugging Face, copie `tokenizer.json` et `tokenizer_config.json` depuis le dossier des poids
dans `outils/tokenizer/`. Vérifie la licence du modèle avant de versionner ces fichiers.

### 2.4 Tester la connexion

```bash
curl -s -H "Authorization: Bearer $CLE_API" "$URL/v1/models"
python outils/config_serveur.py --url "$URL" --modele "$MODELE" --metriques "$METRIQUES" > /dev/null
```

**Tu dois voir** le modèle, puis des lignes `[config]` :

```
[config] empreinte de la cible : vllm-<version>-tp8-ep-<hash> -> TP 8, PP 1, DP 1, EP oui
[config] http://10.0.0.5:8000/metrics : vLLM <version> ; cache de préfixes False ; jetons spéculatifs (métriques) 3 ; /server_info absent (VLLM_SERVER_DEV_MODE non activé)
```

(exemple pour un serveur TP 8 avec parallélisme d'experts et 3 jetons spéculatifs ; « jetons spéculatifs 0 »
= pas de décodage spéculatif).

Contrôle que le **cache de préfixes** et le **parallélisme** sont ceux de la mesure que tu vas faire.
- **Empreinte absente, « pas de réponse »** : nom du modèle, clé ou cible à corriger avant de continuer.
- **Empreinte absente sans autre message** : version de vLLM qui ne la produit pas, `--fingerprint-mode none`,
  ou passerelle qui ne la transmet pas ; vise vLLM directement, ou note la commande de lancement avec tes résultats.
- **« TP 1, PP 1, DP 1, EP non (ou masqué…) »** : vrai pour un serveur sans parallélisme, mais aussi ce
  qu'affiche un serveur lancé en `--fingerprint-mode hash` : vérifie la commande de lancement.

---

## 3. Charge synthétique

vLLM lancé **avec** `--no-enable-prefix-caching` (§ 1.2).

### 3.1 Balayer

```bash
./balayage.sh essai-r1 "1 2 4 8 16 32 64"
```

- Une **étiquette** par essai (`essai-r1`) ; le suffixe `-r1`, `-r2`… marque les répétitions d'une même
  configuration, que le dépouillement regroupe. Une étiquette déjà utilisée est refusée (`ECRASER=1` la remplace).
- Une liste de **concurrences** : un palier par valeur.
- Réglages par variables, devant la commande :

| Variable | Défaut | Sens |
|---|---|---|
| `ISL` / `OSL` | 2048 / 256 | jetons d'entrée / de sortie par requête (sortie imposée : `min_tokens` + `ignore_eos`) |
| `DUREE` | vide | secondes par palier ; vide = 8 × concurrence requêtes (au moins 16) |
| `RACINE` | `resultats` | dossier des résultats |
| `TEMPERATURE` | 0.7 | fixée pour que le décodage spéculatif soit comparable d'un essai à l'autre |
| `EXTRA` | vide | options AIPerf en plus, ex. `--use-server-token-count` |

Exemple pour une mesure de référence : `ISL=32768 OSL=512 DUREE=900 ./balayage.sh conf-a-r1 "48 52 56 60 64"`.

**Tu dois voir** une ligne par palier avec sa durée, puis `Terminé`. Des requêtes en erreur n'arrêtent pas
AIPerf : vérifie la colonne `err` au dépouillement.

### 3.2 Dépouiller

```bash
python depouiller.py resultats --cible 30 --gpu 8
```

- `--cible` : interactivité visée, en jetons/s par requête.
- `--gpu` : nombre de GPU derrière la cible (toutes répliques comprises).
- `--exiger-metriques` : refuse un palier sans métriques du serveur.
- `--ttft-max <ms>` : refuse un palier dont le TTFT P95 dépasse ce seuil.

| Colonne | Sens |
|---|---|
| débit/GPU | jetons de sortie par seconde, toutes requêtes confondues, ÷ nombre de GPU |
| inter P10 | interactivité au 10e centile : 90 % des requêtes ont écrit au moins à cette vitesse |
| TTFT P95 ms | attente avant le premier jeton, 95e centile |
| préempt | requêtes suspendues faute de mémoire ; doit être 0 |
| err | requêtes en erreur ; doit être 0 |
| cache | part des jetons d'entrée trouvés dans le cache de préfixes ; « — » ou ≈ 0 ici (cache désactivé) |
| MTP | taux d'acceptation du décodage spéculatif ; « — » sans décodage spéculatif |
| sortie moy | jetons de sortie moyens par requête ; doit valoir `OSL` |

Puis le **point retenu** : le palier au débit le plus élevé qui respecte la cible, sans erreur ni préemption
(médiane des répétitions), et une **estimation interpolée exactement à la cible**. Lis les avertissements :
« plus haut palier testé » veut dire qu'il faut ajouter des paliers au-dessus. Tableau complet :
`resultats/tableau.csv`.

---

## 4. Agents de test

**Redémarre vLLM avec le cache de préfixes activé** (sans `--no-enable-prefix-caching`, § 1.2), vérifie
`cache de préfixes True` dans la ligne `[config]` du § 2.4, puis :

```bash
cd agents_test
python lanceur.py --url "$URL" --modele "$MODELE" --metriques "${METRIQUES:-$URL/metrics}" \
  --paliers "1 2 4 8" --duree 900 --prechauffe 300 --echelle 1 --etiquette conf-a-r1
python depouiller_agents.py resultats-agents --cible 30 --gpu 8
```

| Option | Sens |
|---|---|
| `--paliers` | nombres de tâches simultanées ; chaque palier tient `--duree` secondes en alternant les trois profils |
| `--prechauffe` | secondes exclues de la mesure en début de palier (au moins la durée d'une tâche) ; doit être < `--duree` |
| `--echelle` | multiplie toutes les tailles : 1 = contextes jusqu'à ≈ 135 000 jetons ; 0.2 pour un petit modèle à 32 768 de contexte |
| `--agents` | sous-ensemble de `revue_code,pentest,rapport` |
| `--mode` | `scripte` (défaut, mesure : outils et longueurs imposés) ou `autonome` (vérifie l'appel d'outils, ne mesure pas) |
| `--sortie` | dossier des résultats (défaut `resultats-agents`) |

La clé est lue dans `BANC_CLE_API` si elle existe, sinon dans `CLE_API`. Le dépouillement donne une ligne « tous » par palier
puis une par profil (débit/GPU, inter P10, TTFT P95, part du cache, durée des tâches, préemptions, MTP,
cache KV maximal), et le palier retenu. **À vérifier** : la colonne cache nettement au-dessus de 0
pour `pentest` et `rapport`. Tableau complet : `resultats-agents/tableau_agents.csv`.

Pour une première vérification rapide : `--paliers "1 2" --duree 300 --prechauffe 60 --echelle 0.2`.

---

## 5. À travers LiteLLM (optionnel)

Même balayage, en visant LiteLLM et en donnant l'URL des métriques de vLLM :

```bash
URL=http://<vm>:4000 MODELE=<alias> CLE_API=<clé LiteLLM> METRIQUES=http://<hôte>:8000/metrics \
  ./balayage.sh conf-a-litellm-r1 "1 4 16"
```

Compare au même balayage direct sur vLLM : l'écart de TTFT est la latence ajoutée par la passerelle.
**Vérifie** que la colonne « sortie moy » du dépouillement vaut `OSL` : sinon LiteLLM ne transmet pas `min_tokens` /
`ignore_eos` et les deux mesures ne sont pas comparables (non vérifié).

Plusieurs répliques derrière LiteLLM : `METRIQUES` reçoit une URL par réplique, `--gpu` le total des GPU.
Mesure les répliques **ensemble**, comme elles serviront.

---

## 6. Protocole de mesure conseillé

Une passe courte ne suffit pas pour retenir un chiffre. Pour chaque configuration de serveur à mesurer :

1. **Passe grossière** (charge synthétique) : paliers larges, 2 min chacun, pour situer le coude —
   `DUREE=120 ./balayage.sh conf-a-grossier "1 2 4 8 16 32 48 64 96 128 192 256"`.
2. **Passe fine** : 4 ou 5 paliers resserrés (≈ 5 à 15 % d'écart, ex. `"48 52 56 60 64"`) autour du coude (dernier palier qui tient la cible,
   premier qui ne la tient pas), **15 min** chacun (`DUREE=900`), **trois répétitions** (`conf-a-r1`, `-r2`, `-r3`).
   Retiens la médiane et l'écart min–max, et l'estimation interpolée à la cible. Plus de 10 % d'écart entre
   répétitions (seuil indicatif) : machine perturbée, à refaire.
3. **Sans décodage spéculatif** (si le modèle en utilise) : relance vLLM sans `--speculative-config` et refais
   la passe fine dans une autre racine (`RACINE=resultats-sansmtp`). Le gain du décodage spéculatif dépend du
   texte et de la température : note le taux d'acceptation (colonne MTP) à côté de chaque point.
4. **Agents de test** (cache activé) : paliers jusqu'au-delà du palier retenu, 15 min, préchauffe 300 s,
   trois répétitions.
5. **À travers LiteLLM** : un balayage court (§ 5), pour la latence ajoutée.
6. **Reprise** : arrête une réplique, relance-la, et mesure le temps jusqu'à la première réponse
   (`curl -s -H "Authorization: Bearer $CLE_API" "$URL/v1/models"` toutes les 10 s) ; pas de balayage pendant
   ce test.
7. **Comparer** les configurations : débit synthétique avec et sans MTP, débit des agents et taux de cache,
   TTFT P95, cache KV maximal, temps de reprise. Le choix de la configuration et de la cible ne relève pas du banc.

Ordre de grandeur, par configuration et par contexte : ≈ 4 h pour les passes grossière et fine (5 paliers ×
15 min × 3), ≈ 5 h 30 pour les agents avec 7 paliers (7 × 15 min × 3).

---

## 7. Interface (optionnel)

```bash
python interface/serveur.py            # option --port, défaut 8765
```

Ouvre **l'adresse exacte affichée** (elle contient un jeton d'accès). Sur une VM distante :
`ssh -L 8765:127.0.0.1:8765 <vm>`, puis la même adresse sur ton poste.

- **Lancer** : champs de connexion (mêmes sens qu'au § 2.2), bouton « Tester la connexion » qui affiche la
  configuration du serveur (version, cache de préfixes, TP/PP/DP/EP…), puis charge synthétique ou agents.
- **Essais** : journal en direct, arrêt. **Résultats** : tableaux, point retenu, courbe.

Les champs ne sont pas enregistrés, et l'interface n'hérite pas des variables de `cible.env`. Un seul essai
à la fois. Détails : `interface/README.md`.

---

## 8. Ce que chaque essai enregistre

```
resultats/<étiquette>/config_serveur.json     # configuration du serveur au début de l'essai
resultats/<étiquette>/c<N>/                    # sorties AIPerf du palier de concurrence N (+ c<N>.log)
agents_test/resultats-agents/<étiquette>/config_serveur.json     # lanceur lancé depuis agents_test/
agents_test/resultats-agents/<étiquette>/n<N>/   # appels.jsonl, taches.jsonl, palier.json (dont métriques vLLM)
```

`config_serveur.json` contient : version de vLLM, modèle et contexte maximal, configuration du cache,
empreinte et parallélisme décodé, nombre de jetons spéculatifs (compté dans `/metrics`), et, seulement si
vLLM est lancé avec `VLLM_SERVER_DEV_MODE=1`, la méthode de décodage spéculatif et l'ordonnanceur
(`max_num_seqs`, `max_num_batched_tokens`). **Ce mode est déconseillé** : il ouvre aussi des points
d'accès de contrôle (dont le remplacement des poids) qui répondent sans clé
([enregistrement](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/launchers/api_server/routers.py#L46-L53),
[routes](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/rl/online/api_router.py)). La page sécurité de vLLM présente certaines de ces
routes comme ouvertes même hors mode dev ; dans le doute, traite tout port vLLM comme sensible. Si tu l'utilises : serveur de banc seulement, port ouvert au seul client de
test, jamais en production.

---

## 9. Dépannage

| Symptôme | Cause probable | Que faire |
|---|---|---|
| `ÉCHEC à la concurrence N` | voir `resultats/<étiquette>/c<N>.log` | souvent tokenizer introuvable (§ 2.3) ou cible injoignable (§ 2.4) |
| `HFValidationError` ou tokenizer introuvable dans le journal AIPerf | AIPerf lancé à la main avec `HF_HUB_OFFLINE=1`, ou `TOKENIZER` = nom Hugging Face absent du cache local | passe par `balayage.sh` avec le tokenizer de `outils/tokenizer/` (§ 2.3) |
| colonnes préempt / cache / MTP vides | `/metrics` non lu | vérifie `METRIQUES` (obligatoire via LiteLLM) |
| `err` > 0 | requêtes refusées ou expirées | clé, nom du modèle, `--max-model-len`, `--max-num-seqs` |
| débit des agents « — » | fenêtre de mesure vide | `--prechauffe` plus court que `--duree`, paliers assez longs |
| cache ≈ 0 pour les agents | cache de préfixes désactivé | redémarre vLLM sans `--no-enable-prefix-caching` |
| TP/PP/DP « n/d » dans l'interface | pas d'empreinte | modèle et clé corrects ; `--fingerprint-mode` par défaut ; viser vLLM directement |
| « Un essai est en cours : pas de complétion d'un jeton, donc pas d'empreinte » | test de connexion pendant un essai | normal : la complétion d'un jeton fausserait la mesure |

---

## 10. Vérifier la chaîne sans GPU

Un faux serveur compatible OpenAI permet de tester les scripts sans modèle (les chiffres ne mesurent rien) :

```bash
python outils/serveur_factice.py --port 8001 &      # à arrêter ensuite : kill %1
cd agents_test && python lanceur.py --url http://127.0.0.1:8001 --paliers "1 2" --duree 20 \
  --prechauffe 2 --echelle 0.05 --facteur-latence 0.02 --etiquette essai --sortie /tmp/essai
python depouiller_agents.py /tmp/essai --cible 30 --gpu 1
```

---

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `balayage.sh`, `depouiller.py` | charge synthétique (AIPerf) et son dépouillement |
| `agents_test/agents.py` | les trois agents de test et leurs outils simulés |
| `agents_test/lanceur.py`, `depouiller_agents.py` | lancement par paliers, dépouillement |
| `agents_test/commun.py`, `metriques.py` | appel mesuré au modèle (streaming), relevé de `/metrics` |
| `outils/config_serveur.py` | relevé de la configuration des serveurs vLLM |
| `outils/telecharger_tokenizer.sh`, `cible.env.exemple` | tokenizer par défaut, modèle de réglages |
| `outils/serveur_factice.py` | faux serveur pour vérifier la chaîne |
| `interface/` | interface locale (React compilée + serveur Python) |

Les agents de test n'utilisent aucune donnée interne. Les journaux contiennent des nombres (heures, jetons,
durées), les étiquettes, les noms d'agents et les messages d'erreur ; aucun contenu d'invite ni de réponse.

## Sources

- vLLM, code au commit `7f855f6` :
  [empreinte `system_fingerprint`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/serve/utils/fingerprint.py) ·
  [`--api-key`, `--fingerprint-mode`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/common/cli_args.py) ·
  [contrôle de la clé](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/serve/middleware/authenticate.py) ·
  [routes de développement](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/launchers/api_server/routers.py) ·
  [`/server_info`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/serve/dev/server_info/api_router.py) ·
  [métriques du décodage spéculatif](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/v1/spec_decode/metrics.py) ·
  [`VLLM_SERVER_DEV_MODE`, `DO_NOT_TRACK`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/envs.py)
- vLLM, documentation : [métriques](https://docs.vllm.ai/en/stable/usage/metrics/) ·
  [sécurité](https://docs.vllm.ai/en/latest/usage/security.html) ·
  [appel d'outils](https://docs.vllm.ai/en/stable/features/tool_calling.html)
- [AIPerf 0.12.0](https://pypi.org/project/aiperf/0.12.0/) ·
  [balayage de concurrence (NVIDIA)](https://docs.nvidia.com/nim/benchmarking/llm/latest/step-by-step.html)
- LiteLLM : [fournisseur vLLM](https://docs.litellm.ai/docs/providers/vllm) · [configuration du proxy](https://docs.litellm.ai/docs/proxy/configs)
