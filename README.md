# banc-gpu

Banc de mesure de capacité pour un modèle servi par vLLM : pour une installation donnée, combien de
requêtes simultanées (ou de tâches d'agents) elle tient, et à quel débit par GPU, tant que chaque
requête reçoit ses jetons à une vitesse visée (l'**interactivité**, en jetons/s par requête).

Deux familles de mesures :

- les **stress tests** : le banc génère lui-même la charge, par paliers croissants, jusqu'à la rupture ;
  ils donnent la **capacité** de l'installation ;
- l'**observation** : le banc ne génère rien et n'a rien à lancer ; Prometheus collecte en continu les métriques
  de vLLM (et de LiteLLM) pendant que tes vrais agents tournent, et le banc l'interroge pour dire si l'installation
  **tient la charge réelle**, par fenêtre, par période et par agent, sur n'importe quelle plage passée.

| Mode | Outil | Charge | Cache de préfixes de vLLM | Ce qu'il donne | § |
|---|---|---|---|---|---|
| **Stress test synthétique** | `balayage.sh` (AIPerf) | requêtes à longueurs fixes | **désactivé** | plancher prudent | 3 |
| **Stress test AgentX** | `agentx.sh` (AIPerf, scénario `inferencex-agentx-mvp`) | sessions publiques d'agents de code rejouées | activé | mesure comparable aux résultats AgentX d'InferenceX | 4 |
| **Stress test agents de test** | `agents_test/` | trois profils d'agents simulés | activé | charge d'agents proche de vos usages | 5 |
| **Observation** | `observation/interroger.py` + Prometheus | tes agents, en usage réel | tel que configuré | « ça tient ? », par fenêtre, période, agent | 6 |

Écrit d'après le code de **vLLM au commit [`7f855f6`](https://github.com/vllm-project/vllm/tree/7f855f605892656ebf7265642e5d22a34a6ae731)** et testé avec **AIPerf 0.12.0** contre le serveur
factice (§ 12). Contre un vrai vLLM, seulement sur un Mac (vllm-metal, vLLM 0.33.0) : connexion à travers
LiteLLM, empreinte, `/metrics`, chargement des traces AgentX ; **pas encore sur un serveur GPU**. La vue par agent
(§ 6.5, façon 2) a été vérifiée avec un vrai LiteLLM (1.109.1) lors de la validation. Contrôle les sorties réelles au § 2.4
(sources en fin de document).

Les agents de test sont trois profils (`revue_code` avec sous-agents, `pentest` pas à pas, `rapport`).
Leurs outils sont simulés : ils renvoient du texte fabriqué et n'agissent sur aucun système.

---

## 1. Requis

### 1.1 Ce qui doit déjà être en place

| Requis | Détail | Vérification |
|---|---|---|
| **vLLM installé et le modèle servi** | API compatible OpenAI sur `http://<hôte>:<port>`. Le banc ne démarre pas vLLM. | `curl -s http://<hôte>:8000/v1/models` liste le modèle |
| **`/metrics` de vLLM joignable** depuis la machine cliente | Actif par défaut ; il donne préemptions, cache, décodage spéculatif, occupation du cache KV. Il n'exige pas la clé d'API (§ 1.2). | `curl -s http://<hôte>:8000/metrics \| grep -c '^vllm:'` affiche un nombre supérieur à 0 |
| **LiteLLM installé** (optionnel) | Seulement pour mesurer à travers la passerelle. Modèle déclaré en `hosted_vllm/<nom>` avec `api_base: http://<hôte>:8000/v1` et, si vLLM a une clé, `api_key` ([LiteLLM, vLLM](https://docs.litellm.ai/docs/providers/vllm) ; [configuration](https://docs.litellm.ai/docs/proxy/configs)). | `curl -s -H "Authorization: Bearer <clé>" http://<vm>:4000/v1/models` |
| **Prometheus** (observation seulement) | Serveur à part qui collecte `/metrics` de vLLM (et de LiteLLM) et garde l'historique ; vLLM n'en fournit pas. Un seul binaire ([téléchargement](https://prometheus.io/download/)), ou l'image Docker `prom/prometheus`. Configuration fournie : `outils/prometheus.yml.exemple` (§ 6.2). | `curl -s http://<hôte>:9090/-/ready` |
| **Python 3.11 à 3.13** sur la machine cliente | Exigé par AIPerf 0.12. | `python3 --version` |
| **bash** | Testé avec bash 5 ; écrit pour bash 3.2 (macOS), vérifié seulement avec un bash 3.2 compilé sous Linux. | `bash --version` |
| **Réseau** | La machine cliente atteint vLLM (et LiteLLM). Aucun accès Internet n'est nécessaire une fois le tokenizer copié dans `outils/tokenizer/` (§ 2.3). | — |

La **machine cliente** est celle qui lance les mesures : le même poste que vLLM pour un essai local, ou
une VM sur le même réseau privé que les serveurs GPU. Ne lance pas les clients sur les serveurs mesurés.

### 1.2 Vérifier la configuration du vLLM en place

Le banc ne lance jamais vLLM : il mesure celui qui tourne. Mais la façon dont ce vLLM a été lancé change le sens
de certaines mesures. Avant chaque mode, vérifie ses options ; le test de connexion (§ 2.4) affiche le cache de
préfixes, la version et le parallélisme du vLLM en place.

En pratique :

- **Observation** : rien à changer ; `--custom-histogram-buckets` rend seulement la part à la cible exacte
  (sinon interpolée), à ajouter au prochain redémarrage prévu.
- **AgentX, agents de test** : le vLLM de production convient en général (cache de préfixes activé, le défaut) ;
  vérifie `--max-model-len` et `--max-num-seqs` ci-dessous.
- **Synthétique** : seul mode qui demande un **redémarrage** dédié, avec `--no-enable-prefix-caching` ; sinon le
  débit mesuré est gonflé par le cache. Hors d'un serveur de production en service, donc.

Options qui comptent pour le banc :

| Option de `vllm serve` | Synthétique | AgentX | Agents de test | Observation | Pourquoi |
|---|---|---|---|---|---|
| `--served-model-name <nom>` | conseillé | conseillé | conseillé | — | nom court à donner au banc (`MODELE`) |
| `--no-enable-prefix-caching` | **requis** | **ne pas mettre** | **ne pas mettre** | selon ta production | AIPerf réutilise ses invites synthétiques : avec le cache, le débit serait gonflé. Les agents relisent leur contexte : le cache fait partie de ce qu'on mesure. |
| `--max-model-len` | ≥ entrée + sortie + gabarit de conversation (ex. 2 048 + 256, plus une marge) | ≥ le plus long contexte des traces : jusqu'à ≈ 990 000 jetons d'entrée par requête (jeu complet), 256 000 (variante `256k`) ; sinon `CONTEXTE_MAX` (§ 4.2) | ≥ ≈ 135 000 × `--echelle`, plus la sortie | — | AIPerf ne tronque pas les traces AgentX ; les agents de test atteignent ≈ 135 000 jetons à l'échelle 1 (estimé avec le vocabulaire Qwen) |
| `--max-num-seqs` | ≥ la plus haute concurrence | ≥ sessions × sous-agents en parallèle (INFÉRÉ ; garde une marge) | ≥ le plus haut palier × 3 (`revue_code` lance 3 sous-agents) | — | sinon les requêtes en trop attendent |
| `--enable-prompt-tokens-details` | — | — | conseillé | — | chaque réponse indique les jetons lus en cache |
| `--enable-auto-tool-choice --tool-call-parser <parseur>` | — | — | mode `autonome` seulement | — | appel d'outils choisi par le modèle |
| `--custom-histogram-buckets '…'` | — | — | — | **conseillé** | bornes de l'histogramme du temps par jeton à tes cibles : interactivité exacte (§ 6.1) |
| `--api-key <clé>` | optionnel | optionnel | optionnel | optionnel | à donner au banc (`CLE_API`) ; voir la mise en garde ci-dessous |
| `--fingerprint-mode` | défaut (`full`) | défaut | défaut | défaut | le banc lit le parallélisme (TP, PP, DP, EP) dans l'empreinte des réponses ; en mode `hash`, il est masqué |

**`--api-key` ne suffit pas à protéger vLLM.** Elle ne s'applique qu'aux chemins `/v1`, `/v2`, `/inference`
et `/cohere` ; les autres restent ouverts, dont `/metrics` et `/invocations`, qui donne la même inférence
que `/v1` ([vLLM, `cli_args.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/common/cli_args.py#L337-L347) ;
[sécurité vLLM](https://docs.vllm.ai/en/latest/usage/security.html#api-key-authentication-limitations)).
Ouvre le port de vLLM au seul client de test (pare-feu, réseau privé). Côté serveur, `DO_NOT_TRACK=1`
coupe l'envoi de statistiques d'usage de vLLM ([`envs.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/envs.py), [`usage_lib.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/usage/usage_lib.py)).

Changer le cache de préfixes impose de **redémarrer vLLM** entre la charge synthétique et les autres modes.

### 1.3 Ce que le banc installe

```bash
git clone https://github.com/Shevehs/banc-gpu.git ~/banc && cd ~/banc
python3 -m venv .venv-aiperf && source .venv-aiperf/bin/activate
pip install aiperf==0.12.0 "openai>=1.40" httpx
aiperf --version && python -c "import openai, aiohttp; print('ok')"
```

AIPerf installe aussi `aiohttp` (interface) et la commande `hf` (tokenizer). Sans accès Internet,
`pip` doit pointer vers un miroir de paquets.

**Python trop ancien** (ex. 3.9 livré avec macOS), **à la place** des lignes `python3 -m venv` et `pip install`
ci-dessus (si `.venv-aiperf` existe déjà, supprime-le d'abord) : sans toucher au Python du système,
[uv](https://docs.astral.sh/uv/guides/install-python/) télécharge un Python 3.12 pour cet environnement seulement :

```bash
python3 -m pip install --user uv            # ou l'installeur de https://docs.astral.sh/uv/getting-started/installation/
uv venv --seed --python 3.12 .venv-aiperf   # --seed : met pip dans l'environnement
source .venv-aiperf/bin/activate && python --version
pip install aiperf==0.12.0 "openai>=1.40" httpx
```

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

Remplis `cible.env` (garde les `export`), puis charge-le dans chaque nouveau terminal : `source cible.env`.

Pourquoi `METRIQUES` doit viser vLLM : les stress tests y lisent l'état du moteur, que seul vLLM connaît.
Si ce sont les métriques de LiteLLM (ou rien), sont perdus : les **préemptions** (un palier où vLLM manque de
mémoire peut alors être retenu comme sain), le cache KV maximal, la file d'attente du moteur, le cache de
préfixes et l'acceptation du décodage spéculatif ; les colonnes restent vides. Restent valables le débit,
l'interactivité, le TTFT et les erreurs, mesurés côté client. Au lancement de chaque essai, une ligne
`[config] ATTENTION : … n'est pas le /metrics de vLLM` signale l'erreur (LiteLLM répond « HTTP 401 : clé
demandée », ou sans aucune métrique `vllm:`).

| Variable | Sens | Exemple |
|---|---|---|
| `URL` | **cible** des requêtes : vLLM directement (mesure des GPU) ou LiteLLM (mesure à travers la passerelle) | `http://10.0.0.5:8000` |
| `MODELE` | nom servi par vLLM (`--served-model-name`) ou alias LiteLLM | `banc` |
| `METRIQUES` | adresse `/metrics` **de vLLM** (jamais de LiteLLM ni de Prometheus), une par réplique, séparées par des espaces. Vide = `URL/metrics`, qui ne convient que si `URL` vise vLLM directement. Si `URL` vise LiteLLM, donne l'adresse de vLLM | `http://10.0.0.5:8000/metrics` |
| `CLE_API` | clé de la cible (`--api-key` de vLLM, ou clé LiteLLM). Envoyée à la cible seulement, jamais journalisée | — |
| `TOKENIZER` | vide = `outils/tokenizer/` (§ 2.3) ; sinon chemin absolu d'un dossier qui contient `tokenizer.json` (sans copie), ou nom Hugging Face (AIPerf le télécharge ; les agents de test, qui ne lisent qu'un dossier local, estiment alors les tailles) | vide |

**En entreprise seulement** (sur un poste local, rien à faire) :
- **Proxy** : si `HTTP_PROXY` ou `HTTPS_PROXY` sont définis, le banc les ignore et joint les serveurs
  directement, comme AIPerf : sa clé d'API ne sort pas du réseau privé par le proxy, et la mesure n'inclut pas
  la latence du proxy. Si la machine cliente ne peut joindre les serveurs **que** par un proxy (rare sur un
  réseau privé), le banc ne fonctionnera pas tel quel.
- **Certificat interne** : si la cible est en `https://` avec un certificat signé par l'autorité de
  certification de l'entreprise, indique le fichier de cette autorité : `export SSL_CERT_FILE=/chemin/ac.pem`
  (lu par AIPerf, les scripts et les agents de test).

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
dans `outils/tokenizer/`. **Sans copie** : indique dans `cible.env` le dossier qui contient déjà `tokenizer.json`,
par exemple celui que vLLM a téléchargé :
`TOKENIZER=$HOME/.cache/huggingface/hub/models--<organisation>--<modèle>/snapshots/<hash>` (chemin absolu ; le
dossier, pas le fichier ; celui du modèle réellement servi). Vérifie la licence du modèle avant de versionner ces fichiers.

### 2.4 Tester la connexion

```bash
curl -s -H "Authorization: Bearer $CLE_API" "$URL/v1/models"
python outils/config_serveur.py --url "$URL" --modele "$MODELE" --metriques "$METRIQUES" > /dev/null
```

**Tu dois voir** le modèle, puis des lignes `[config]` :

```
[config] empreinte de la cible : vllm-<version>-tp8-ep-<hash> -> TP 8, PP 1, DP 1, EP oui
[config] GPU déduits : 8 (TP 8 × PP 1 × DP 1 × 1 serveur(s) vLLM)
[config] http://10.0.0.5:8000/metrics : vLLM <version> ; cache de préfixes False ; jetons spéculatifs (métriques) 3 ; /server_info absent ou non lu (VLLM_SERVER_DEV_MODE non activé ?)
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

## 3. Stress test synthétique

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
python depouiller.py resultats --cible 30
```

- `--cible` : interactivité visée, en jetons/s par requête.
- `--gpu` : nombre de GPU derrière la cible (toutes répliques comprises). **Facultatif** : sans lui, le
  dépouillement le déduit du `config_serveur.json` de chaque essai (TP × PP × DP lus dans l'empreinte, × nombre
  de serveurs vLLM reconnus dans `METRIQUES`) et l'affiche (`GPU : <essai> : 8 (TP 8 × PP 1 × DP 1 × 1 serveur(s)
  vLLM)`) ; un ATTENTION signale des répétitions d'une même configuration aux nombres différents. **Donne-le**
  si l'empreinte manque (le script le demande, raison à l'appui), si elle n'a aucun degré de parallélisme alors
  que le serveur a plusieurs GPU (`--fingerprint-mode hash` les masque ; un ATTENTION le rappelle ; sur un Mac,
  cet ATTENTION est attendu), si des répliques diffèrent, avec le parallélisme de contexte (PCP, absent de
  l'empreinte ; lu seulement si `/server_info` existe), ou avec un DP réparti en un serveur d'API par rang
  (chaque empreinte porte le DP global : le compte serait trop haut ; INFÉRÉ). Donné, il prévaut ; s'il
  contredit l'empreinte, un ATTENTION le signale.
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

## 4. Stress test AgentX

Rejoue les sessions d'agents de code (Claude Code, sous-agents compris) publiées par SemiAnalysis, avec le
scénario verrouillé d'AIPerf `inferencex-agentx-mvp`, celui des mesures AgentX d'InferenceX
([méthode](https://inferencex.semianalysis.com/blog/agentic-benchmark-agent-benchmark-guide)). Le résultat
se compare donc aux exports AgentX publiés (mêmes traces, même scénario ; moteur et matériel différents).
Les contextes sont longs (médiane ≈ 142 000 jetons d'entrée, d'après
[InferenceX](https://inferencex.semianalysis.com/blog/agentic-benchmark-agent-benchmark-guide)) et ne sont pas tronqués : la mesure complète se
fait sur serveurs GPU : sur un petit modèle (Mac, 32 768 jetons), aucune trace ne tient (§ 4.2).

Ce que le scénario impose (DOCUMENTÉ, code d'AIPerf 0.12.0, `aiperf/common/scenario/inferencex_agentx_mvp.py`,
et message d'AIPerf à l'exécution) : streaming ; sortie à la longueur de la trace (`ignore_eos`, ajouté
d'office) ; **au moins 900 s par palier** (1 800 par défaut) ; aucune troncature des entrées ; délais entre
tours rejoués ; préfixe du premier tour brouillé, pour qu'un essai ne profite pas du cache du précédent ; jeu de
traces **épinglé** (`semianalysisai/cc-traces-weka-062126`) pour que l'essai soit marqué valide.

Les traces sont publiées sur Hugging Face : par défaut, `agentx.sh` les **télécharge lui-même** au premier essai
(≈ 1,85 Go, accès Internet requis), et l'essai est marqué valide. Sans Internet, voir § 4.1.

vLLM lancé avec le cache de préfixes **activé** (les traces portent les empreintes des blocs de cache, pour
reproduire leur réutilisation). Puis :

```bash
DUREE=1800 ./agentx.sh agentx-r1 "1 2 4 8"
python depouiller.py resultats-agentx --cible 30
```

| Variable | Défaut | Sens |
|---|---|---|
| `DUREE` | 1800 | secondes par palier ; au moins 900 |
| `AGENTX_DOSSIER` | vide | dossier local des traces (sans accès Internet, § 4.1) ; l'essai est alors marqué `submission_valid=false` (AIPerf ne peut pas vérifier le jeu) |
| `AGENTX_VARIANTE` | vide | vide : jeu complet `semianalysisai/cc-traces-weka-062126` ; `256k` : `…-062126-256k`, où SemiAnalysis a retiré les requêtes de plus de 256 000 jetons (pour un `--max-model-len` d'environ 256 000 ; accepté par le scénario). D'après la description du jeu dans AIPerf (`plugins.yaml`) |
| `RACINE` | `resultats-agentx` | dossier des résultats |
| `ESSAI_COURT` | vide | `1` = paliers de moins de 900 s, pour vérifier la chaîne seulement (non comparable) |
| `CONTEXTE_MAX` | vide | jetons ; AIPerf écarte les traces dont le contexte de pointe dépasse cette valeur (`--max-context-length`) |
| `AIPERF_DATASET_CONFIGURATION_TIMEOUT` | 1800 | secondes accordées au chargement des traces (AIPerf : 300 par défaut, trop court pour les 393 traces) ; `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT` est porté à au moins cette valeur, comme AIPerf l'exige |

Les paliers sont des nombres de **sessions rejouées en même temps** (`--concurrency` d'AIPerf). Le dépouillement
est celui de la charge synthétique (§ 3.2), sauf la colonne « sortie moy », qui varie ici d'une trace à l'autre.
Durée : 5 paliers × 30 min × 3 répétitions ≈ 7 h 30 par configuration (≈ 3 h 45 à 15 min par palier).

### 4.1 Machine cliente sans accès Internet : copier les traces

Une fois, sur une machine qui a accès à Hugging Face (où les traces sont publiées) :

```bash
./outils/telecharger_traces_agentx.sh traces-agentx     # 393 traces, ≈ 1,85 Go
```

Copie le dossier sur la machine cliente, puis `AGENTX_DOSSIER=traces-agentx ./agentx.sh …`. Le jeu est publié
sous licence Apache-2.0 d'après sa [fiche Hugging Face](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) :
vérifie-la avant de le faire entrer dans ta zone. Non vérifié : que le dépôt range bien une trace par fichier
JSON dans un seul dossier (le script compte les fichiers et le signale).

### 4.2 Contexte plus court que les traces (`CONTEXTE_MAX`)

Le scénario interdit de tronquer une trace (et AIPerf n'a pas d'option pour le faire), mais il accepte d'écarter
celles qui ne tiennent pas (`--max-context-length` d'AIPerf : DOCUMENTÉ par son aide, et sans avertissement du
scénario à l'essai). Utile sur un serveur GPU dont `--max-model-len` est un peu plus court que les plus longues
traces :

```bash
CONTEXTE_MAX=129024 DUREE=1800 ./agentx.sh agentx-128k-r1 "1 2 4 8"   # vLLM : --max-model-len 131072
```

Le filtre porte sur le contexte **maximal** de chaque session. **Pas sur un petit modèle** : à 30 720 jetons
(vLLM à 32 768), aucune des 393 traces ne tient (constaté lors de la répétition sur Mac : « No eligible traces …
scanned 393 »). AIPerf s'arrête alors avant le premier palier, et le script le dit. Sur un Mac, la charge
d'agents se mesure avec les agents de test (§ 5) ; AgentX se mesure sur les serveurs GPU.

Laisse une **marge** sous `--max-model-len` (ici 2 048 jetons) : le filtre compare les longueurs enregistrées
dans les traces, alors que le serveur compte en plus le gabarit de conversation (rôles, séparateurs ; INFÉRÉ).
Une requête dont l'**entrée** dépasse `--max-model-len` est refusée par vLLM ; AIPerf la retire de ses métriques,
et le dépouillement la compte dans `err` et refuse le palier. Si l'entrée tient mais pas la sortie, vLLM ne refuse
pas : il réduit `max_tokens` à la place restante (DOCUMENTÉ par le code de vLLM, `vllm/entrypoints/serve/utils/api_utils.py`, `get_max_tokens`) ;
le dépouillement le signale par un ATTENTION (`osl_mismatch_count` d'AIPerf, colonne `sorties_raccourcies`).

Ce sous-ensemble n'est **pas comparable** aux résultats AgentX publiés (INFÉRÉ : biais vers les sessions plus
courtes) ; note le nombre de traces écartées avec le résultat (journal du palier, `resultats-agentx/<étiquette>/c<N>.log`).

---

## 5. Stress test agents de test

**Redémarre vLLM avec le cache de préfixes activé** (sans `--no-enable-prefix-caching`, § 1.2), vérifie
`cache de préfixes True` dans la ligne `[config]` du § 2.4, puis :

```bash
cd agents_test
python lanceur.py --url "$URL" --modele "$MODELE" --metriques "${METRIQUES:-$URL/metrics}" \
  --paliers "1 2 4 8" --duree 900 --prechauffe 300 --echelle 1 --etiquette conf-a-r1
python depouiller_agents.py resultats-agents --cible 30
```

| Option | Sens |
|---|---|
| `--paliers` | nombres de tâches simultanées ; chaque palier tient `--duree` secondes en alternant les trois profils |
| `--prechauffe` | secondes exclues de la mesure en début de palier (au moins la durée d'une tâche) ; doit être < `--duree` |
| `--echelle` | multiplie toutes les tailles : 1 = contextes jusqu'à ≈ 135 000 jetons ; 0.2 pour un petit modèle à 32 768 de contexte |
| `--agents` | sous-ensemble de `revue_code,pentest,rapport` |
| `--mode` | `scripte` (défaut, mesure : outils et longueurs imposés) ou `autonome` (vérifie l'appel d'outils, ne mesure pas) |
| `--sortie` | dossier des résultats (défaut `resultats-agents`) |
| `--etiqueter-agents` | ajoute `metadata.agent` = nom du profil à chaque requête (vue par agent via LiteLLM, § 6.5) |

La clé est lue dans `BANC_CLE_API` si elle existe, sinon dans `CLE_API` ; `BANC_CLES_AGENTS` donne une clé par
profil (§ 6.5). Le dépouillement donne une ligne « tous » par palier
puis une par profil (débit/GPU, inter P10, TTFT P95, part du cache, durée des tâches, préemptions, MTP,
cache KV maximal), et le palier retenu. **À vérifier** : la colonne cache nettement au-dessus de 0
pour `pentest` et `rapport`. Tableau complet : `resultats-agents/tableau_agents.csv`.

Pour une première vérification rapide : `--paliers "1 2" --duree 300 --prechauffe 60 --echelle 0.2`.

---

## 6. Observation (usage réel)

Tes agents appellent vLLM, directement ou via LiteLLM ; le banc ne génère **aucune** requête et n'a **rien à
lancer**. Le travail se répartit ainsi :

- **vLLM** expose ses compteurs sur `/metrics`, cumulés depuis son démarrage ; il ne garde aucun historique
  ([métriques de vLLM](https://docs.vllm.ai/en/latest/design/metrics.html)) ;
- **Prometheus**, serveur à part, lit `/metrics` toutes les 15 s et garde l'historique
  ([présentation](https://prometheus.io/docs/introduction/overview/)) ; il lit aussi les métriques de LiteLLM pour
  la vue par agent ;
- **`observation/interroger.py`** (ou l'onglet Observation de l'interface) interroge Prometheus sur une plage
  passée, la découpe en fenêtres et calcule, par fenêtre : requêtes terminées, concurrence moyenne (requêtes en
  cours), débit de sortie par GPU, **part des requêtes qui tiennent l'interactivité visée**, TTFT, préemptions,
  requêtes en erreur, cache de préfixes, acceptation MTP, cache KV et file d'attente ; puis un verdict.

Une seule lecture de `/metrics` ne suffirait pas : ses compteurs mélangent tout depuis le démarrage. Il faut la
différence entre le début et la fin de chaque fenêtre : `interroger.py` la demande à Prometheus (valeur à la fin
moins valeur au début, [`offset`](https://prometheus.io/docs/prometheus/latest/querying/basics/#offset-modifier)).
Une fenêtre où le serveur a redémarré est écartée : ses compteurs repartent de zéro et la différence n'a plus de
sens (repérée par [`resets()`](https://prometheus.io/docs/prometheus/latest/querying/functions/#resets) et, quand
vLLM l'expose, par `process_start_time_seconds`). Une fenêtre sans collecte juste avant son début (Prometheus
arrêté) est aussi écartée : `offset` irait chercher une valeur jusqu'à 5 min plus tôt
([lookback](https://prometheus.io/docs/prometheus/latest/querying/basics/#staleness)) et la fenêtre compterait la coupure.

**Sans danger pour les outils ?** L'observation ne change rien à tes agents. Mais si tu montes leur nombre pour
chercher la limite, leurs outils agissent pour de vrai à chaque fois : champ de tir isolé ou outils simulés.

### 6.1 Régler vLLM pour une interactivité exacte

L'interactivité vient de l'histogramme `vllm:request_time_per_output_token_seconds` : temps moyen entre deux
jetons de chaque requête, premier jeton exclu ([`stats.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/v1/metrics/stats.py),
[`loggers.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/v1/metrics/loggers.py#L957-L963)), même définition que l'interactivité des stress
tests (même définition côté seuil ; voir les différences ci-dessous). Ses bornes par défaut (10, 25, 50, 75 ms…) ne tombent pas sur 33 ms (30 jetons/s) : la part à la cible
serait interpolée ([`buckets.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/v1/metrics/buckets.py#L84-L92)). Ajoute des bornes à
1/30, 1/50, 1/75 et 1/100 s au lancement de vLLM
([`--custom-histogram-buckets`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/config/observability.py#L72-L80)) :

```bash
vllm serve … --custom-histogram-buckets '{"inter_token_latency": [0.005, 0.01, 0.0133, 0.02, 0.025, 0.0333, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0]}'
```

La famille `inter_token_latency` règle aussi cet histogramme par requête (même famille de bornes dans
`loggers.py`). Le dépouillement marque d'un `*` une part interpolée, et dit « indéterminé » quand la cible tombe
dans un seau qui encadre 90 %.

**Différences avec la mesure des stress tests** (DOCUMENTÉ, `stats.py`, sauf mention) : vLLM horodate dans le
moteur, AIPerf chez le client (réseau et proxy en plus) ; une requête de 0 ou 1 jeton de sortie compte chez vLLM
pour un temps par jeton nul, donc « à la cible », alors qu'AIPerf l'exclut ; une requête abandonnée par le client
n'entre pas dans l'histogramme. Les requêtes en erreur viennent du compteur `http_requests_total` (handlers `/v1/…`,
statuts 4xx et 5xx), ajouté par la bibliothèque prometheus_fastapi_instrumentator que vLLM installe
([`metrics.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/serve/instrumentator/metrics.py#L64-L74)) ;
nom et étiquettes INFÉRÉS de la bibliothèque, à vérifier au premier essai (colonne « err » à « — » sinon).

### 6.2 Installer et lancer Prometheus (une fois)

Sur la machine cliente ou une VM du même réseau privé (pas sur les serveurs mesurés). Deux façons, au choix.

**A. Binaire** (VM Linux ; sans Internet, copie l'archive depuis la
[page de téléchargement](https://prometheus.io/download/), un seul binaire) :

```bash
(cd /tmp && tar xzf <chemin>/prometheus-*.linux-amd64.tar.gz && sudo cp prometheus-*.linux-amd64/prometheus /usr/local/bin/)
cp outils/prometheus.yml.exemple prometheus.yml        # puis remplace les adresses (une ligne par réplique vLLM)
(umask 077; printf '%s' "<clé LiteLLM>" > litellm.cle)  # seulement si tu gardes le bloc litellm
prometheus --config.file=prometheus.yml --storage.tsdb.path=prometheus-data --storage.tsdb.retention.time=30d
```

`prometheus.yml`, `litellm.cle` et `prometheus-data/` sont ignorés par git. Le chemin de `credentials_file` est
relatif au dossier de `prometheus.yml` (constaté lors de la validation, Prometheus 3.7.3).

**B. Docker Compose** (pratique si LiteLLM tourne déjà en Compose). Le `docker-compose.yml` publié par LiteLLM
contient peut-être **déjà** un service `prometheus`, ouvert au réseau sur le port 9090 (constaté par le validateur
sur un miroir du fichier, non sur le dépôt lui-même) : dans ce cas, **modifie-le** pour qu'il ressemble à ce qui
suit, au lieu d'en ajouter un second (Compose refuse deux services ou deux blocs `volumes:` de même nom). Sinon,
ajoute-le :

```yaml
services:
  prometheus:
    image: prom/prometheus:v3.7.3
    ports: ["127.0.0.1:9090:9090"]        # Prometheus ne demande pas de clé : ne l'ouvre pas au réseau
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./litellm.cle:/etc/prometheus/litellm.cle:ro
      - prometheus-data:/prometheus
    command: ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.retention.time=30d"]
volumes:
  prometheus-data:
```

Crée `prometheus.yml` et `litellm.cle` **avant** `docker compose up -d prometheus` (sinon Docker crée des dossiers
vides à leur place et refuse de démarrer). Dans `prometheus.yml`, un vLLM qui tourne sur la machine hôte se joint
par `host.docker.internal:8000` (Docker Desktop), LiteLLM par le nom de son service Compose (`litellm:4000`), et
`credentials_file: /etc/prometheus/litellm.cle`. Le conteneur lit en tant que `nobody` : le fichier de clé doit
être lisible par lui (`printf '%s' "<clé>" > litellm.cle && chmod 644 litellm.cle` ; il devient lisible par tous les
comptes de la machine, d'où l'intérêt d'une clé virtuelle limitée plutôt que la clé maîtresse). Si LiteLLM n'est
pas dans le même Compose (installé par pip sur l'hôte), cible `host.docker.internal:4000` comme vLLM.

Dans tous les cas :

- **Un job par configuration** : toutes les cibles d'un job sont additionnées (toutes les répliques d'une même
  installation). Deux installations à comparer = deux jobs (`job_name: vllm-tep8`, `job_name: vllm-tp4`).
- **Collecte toutes les 15 s** (`scrape_interval`) : les fenêtres doivent durer au moins 4 collectes (1 min) ; si
  tu changes ce réglage, donne la même valeur à `--collecte`.
- **Vérifier** : page `http://<hôte>:9090/targets`, chaque cible doit être « UP » (sinon l'erreur est affichée
  à côté). Puis `curl -s 'http://<hôte>:9090/api/v1/query?query=up'`.
- **Historique** : il commence au lancement de Prometheus. Laisse-le tourner en service ; rétention de 15 jours
  par défaut, 30 avec l'option ci-dessus.
- **Prometheus existant** : si ton équipe d'exploitation a déjà un Prometheus qui collecte vLLM, sers-t'en ;
  donne son URL, le sélecteur de son job (`--vllm 'job="<son job>"'`) et son intervalle de collecte (`--collecte`).
- Testé avec Prometheus 3.7.3 ; les versions plus récentes n'ont pas été essayées.

### 6.3 Interroger

```bash
python observation/interroger.py --prometheus http://127.0.0.1:9090 --depuis 2h --fenetre 5m --cible 30 --gpu 8
```

| Option | Sens |
|---|---|
| `--prometheus` | URL de Prometheus (**obligatoire**) ; jeton dans `BANC_CLE_PROMETHEUS` s'il en exige un |
| `--vllm` | sélecteur des cibles vLLM, défaut `job="vllm"` ; seulement des étiquettes de cible (`job`, `instance`), pas de métrique (`model_name`…) ; toutes les cibles qui correspondent sont additionnées |
| `--depuis` | plage qui se termine maintenant : `30m`, `2h`, `1d` (défaut `2h`) |
| `--debut`, `--fin` | plage passée, heure locale : `--debut "2026-11-14 09:00" --fin "2026-11-14 17:00"` |
| `--fenetre` | durée d'une fenêtre (défaut `5m`) ; au plus 10 000 fenêtres par interrogation |
| `--collecte` | `scrape_interval` de Prometheus (défaut `15s`) |
| `--cible`, `--gpu` | interactivité visée (jetons/s par requête) ; GPU derrière le sélecteur, toutes répliques (**obligatoire** ici : Prometheus ne voit pas l'empreinte) |
| `--ttft-max` | TTFT P95 maximal accepté, en ms (optionnel) |
| `--vue` | `fenetres` (défaut), `periode` (§ 6.4), `agents` (§ 6.5) |
| `--exclure "début/fin"` | plage à écarter, répétable : un stress test sur le même serveur mêlerait sa charge à l'usage réel |
| `--sortie` | écrit les fenêtres dans un CSV |

Une fenêtre **tient** si au moins 90 % des requêtes terminées atteignent `--cible` (même seuil que
l'interactivité P10 des stress tests, § 6.1), sans préemption ni requête en erreur, et, avec `--ttft-max`, si le
TTFT P95 reste sous le seuil ; « indéterminé » si la part à la cible ne peut pas être tranchée (§ 6.1) ;
« aucune requête » si rien n'a été servi. Une fenêtre est **écartée** si une cible vLLM était injoignable au début
ou pendant la fenêtre, si vLLM a redémarré, s'il manque la collecte qui précède la fenêtre ou des collectes dans
la fenêtre (Prometheus arrêté, ou `--collecte` plus
court que son `scrape_interval`) ; elle est marquée « essai en cours » si elle chevauche `--exclure`. Le TTFT et l'interactivité P10 sont des bornes lues dans les
seaux des histogrammes (`≤`, `≥`).

**Chercher la limite avec tes vrais agents** : la colonne `conc moy` donne la concurrence réelle de chaque fenêtre
(requêtes en cours, moyenne de `vllm:num_requests_running`). Le script donne la **plus haute concurrence moyenne
tenue** sur la plage, et prévient si aucune fenêtre plus chargée n'échoue (limite pas encore atteinte). Pour la
chercher, fais monter le nombre d'agents par étapes de quelques fenêtres chacune.

**Limites** (INFÉRÉ sauf mention) :
- une fenêtre couvre de la dernière collecte avant son début à la dernière collecte avant sa fin : ses bords
  sont décalés d'au plus une collecte et demie (une fenêtre sans collecte récente à l'un de ses bords est
  écartée), mais des fenêtres successives s'additionnent sans perte ni double
  compte (vérifié lors de la validation : erreurs et requêtes par agent égales aux compteurs des serveurs) ;
- un redémarrage est repéré par une baisse d'un compteur entre deux collectes (`resets()`) ou par un changement de
  `process_start_time_seconds`. vLLM n'expose ce dernier que sous Linux, et pas quand il lance plusieurs processus
  d'API (`--api-server-count` > 1, le cas par défaut avec `--data-parallel-size` > 1 ; DOCUMENTÉ,
  [`cli/serve.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/cli/serve.py#L107-L130),
  [`v1/metrics/prometheus.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/v1/metrics/prometheus.py#L39-L53)).
  Sans lui, un redémarrage plus court qu'une collecte, dont les compteurs dépassent déjà leur valeur d'avant,
  passe inaperçu : la fenêtre sous-compte, et si ses seaux restent croissants le verdict peut être faux (constaté
  lors de la validation : sinon « indéterminé ») ;
- une série qui disparaît alors que sa cible répond perd ses derniers incréments (rare : les compteurs de vLLM et de
  LiteLLM ne disparaissent pas sans redémarrage) ;
- une série qui apparaît pendant la fenêtre (premier code d'erreur 5xx depuis le démarrage, premier appel d'un
  agent dans LiteLLM) est comptée pour sa valeur entière, à condition que sa cible fût déjà collectée au début de
  la fenêtre ; la fenêtre où la collecte commence est écartée ;
- les requêtes en erreur viennent de `http_requests_total` (handlers `/v1/…`, statuts 4xx et 5xx ; § 6.1) ;
- les jauges (cache KV, file d'attente, concurrence) sont lues toutes les 15 s : une pointe plus courte peut échapper.

### 6.4 Vue par période

```bash
python observation/interroger.py --prometheus http://127.0.0.1:9090 --depuis 1d --fenetre 15m \
  --cible 30 --gpu 8 --vue periode --periode plage
```

Les fenêtres de la plage sont regroupées par plage de 6 h (nuit, matin, après-midi, soir) ou par heure
(`--periode heure`), en heure locale de la machine qui interroge. Le débit d'une période est une moyenne : pour
la pointe, regarde ses fenêtres une à une (`--vue fenetres`).

### 6.5 Vue par agent (via LiteLLM)

vLLM ne sait pas quel agent appelle ; LiteLLM, si, à condition que chaque requête porte l'identité de son agent.
Deux façons de la porter ; dans les deux cas, active d'abord les métriques Prometheus de LiteLLM
([LiteLLM, Prometheus](https://docs.litellm.ai/docs/proxy/prometheus)). Le module `prometheus_client` n'est pas
installé par `pip install "litellm[proxy]"`, et LiteLLM refuse alors de démarrer avec ce réglage (constaté lors
de la validation, LiteLLM 1.109.1) : `pip install prometheus_client` dans son environnement.

```yaml
litellm_settings:
  callbacks: ["prometheus"]
```

**Façon 1 : une clé virtuelle par agent** (rien à changer dans le code des agents ; recommandé en production).
LiteLLM range les clés virtuelles dans une base PostgreSQL : `DATABASE_URL` à définir avant de démarrer LiteLLM
([documentation](https://docs.litellm.ai/docs/proxy/virtual_keys)). Il lui faut aussi Prisma, que
`pip install "litellm[proxy]"` n'installe pas : sans lui, LiteLLM refuse de démarrer avec `DATABASE_URL`, et sans
base, `/key/generate` échoue (constaté lors de la validation, LiteLLM 1.109.1). Avec pip :
`pip install "litellm[proxy,extra_proxy]"` puis, dans le même environnement,
`prisma generate --schema "$(python -c 'import litellm_proxy_extras,os;print(os.path.join(os.path.dirname(litellm_proxy_extras.__file__),"schema.prisma"))')"`
([documentation LiteLLM](https://docs.litellm.ai/docs/troubleshoot/pip_venv_upgrade)). Cette commande installe la
CLI Prisma par npm (Node.js requis) puis télécharge ses moteurs depuis binaries.prisma.sh : sans Internet, il faut
un miroir npm et un miroir des moteurs (constaté lors de la validation : téléchargement des moteurs refusé hors Internet). Le plus simple est de passer par l'installation de LiteLLM de ton équipe si elle a déjà
une base. Crée ensuite une clé par agent, avec la clé maîtresse :

```bash
curl -s -X POST http://<vm>:4000/key/generate -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" -d '{"key_alias": "revue-code"}'      # renvoie la clé de l'agent
```

Chaque agent utilise **sa** clé comme clé d'API (`OpenAI(api_key=…)`, ou la variable de clé de son
framework). Les métriques portent alors l'étiquette `api_key_alias="revue-code"` : `--par api_key_alias` (le
défaut est `metadata_agent`, la façon 2).

**Façon 2 : un champ dans la requête** (pas de base de données ; une ligne dans l'agent). Chaque requête envoie
`"metadata": {"agent": "<nom>"}` dans son corps ; avec le client OpenAI en Python :

```python
client.chat.completions.create(model=…, messages=…, extra_body={"metadata": {"agent": "revue-code"}})
```

et LiteLLM en fait une étiquette :

```yaml
litellm_settings:
  callbacks: ["prometheus"]
  custom_prometheus_metadata_labels: ["metadata.agent"]
```

L'étiquette s'appelle alors `metadata_agent` : `--par metadata_agent`. Si l'agent appelle vLLM sans passer par
LiteLLM, vLLM accepte les champs inconnus et les ignore (trace au niveau debug seulement ; DOCUMENTÉ,
[`protocol.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/entrypoints/serve/engine/protocol.py#L39-L75)).

**Agents de test** : ils savent faire les deux.
- Façon 1 : `BANC_CLES_AGENTS="revue_code=<clé>,pentest=<clé>,rapport=<clé>"` (variable d'environnement, ou
  champ « Clés par agent » de l'interface) ; un profil sans clé garde la clé commune.
- Façon 2 : `lanceur.py --etiqueter-agents` (ou case « Étiqueter les agents ») : `metadata.agent` = nom du profil.

**Collecter, puis interroger** : déclare LiteLLM dans `prometheus.yml` (bloc `litellm` de l'exemple : chemin
**`/metrics/`** avec la barre finale, celui que LiteLLM sert ; clé dans le fichier `litellm.cle` désigné par
`credentials_file` (§ 6.2), car ses métriques exigent une clé par défaut dans les versions récentes : 1.85.0
d'après sa documentation, dès 1.84 d'après son code). Une clé virtuelle limitée vaut mieux que la clé maîtresse
(INFÉRÉ : à vérifier qu'elle suffit pour `/metrics/` sur ta version). Puis :

```bash
python observation/interroger.py --prometheus http://127.0.0.1:9090 --litellm 'job="litellm"' \
  --par metadata_agent --depuis 2h --cible 30 --gpu 8 --vue agents
```

Par agent : requêtes, part des jetons de sortie, jetons d'entrée et de sortie par requête, TTFT P50 et P95,
latence P95 et une vitesse moyenne de génération (jetons de sortie ÷ (latence − TTFT) ; moyenne, pas centile ;
vide si une partie des requêtes n'est pas en streaming ; temps du proxy compris). Les requêtes sans identité
apparaissent sous « (sans identité) » (LiteLLM les étiquette `None`). Noms, types et étiquettes des métriques de LiteLLM vérifiés avec LiteLLM
1.109.1 lors de la validation ; contrôle-les sur ta version au premier essai :

```bash
curl -s -H "Authorization: Bearer <clé>" http://<vm>:4000/metrics/ | grep -E '^litellm_(output_tokens|request_total_latency)' | head
```

---

## 7. Stress test à travers LiteLLM (optionnel)

Même balayage, en visant LiteLLM et en donnant l'URL des métriques de vLLM :

```bash
URL=http://<vm>:4000 MODELE=<alias> CLE_API=<clé LiteLLM> METRIQUES=http://<hôte>:8000/metrics \
  ./balayage.sh conf-a-litellm-r1 "1 4 16"
```

Compare au même balayage direct sur vLLM : l'écart de TTFT est la latence ajoutée par la passerelle.
**Vérifie** que la colonne « sortie moy » du dépouillement vaut `OSL` : sinon LiteLLM ne transmet pas `min_tokens` /
`ignore_eos` et les deux mesures ne sont pas comparables (non vérifié).

Plusieurs répliques derrière LiteLLM : `METRIQUES` reçoit une URL par réplique ; le nombre de GPU déduit les
compte toutes (§ 3.2), en les supposant identiques à celle qui a répondu.
Mesure les répliques **ensemble**, comme elles serviront.

---

## 8. Protocole de mesure conseillé

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
4. **AgentX** (cache activé) : 4 ou 5 niveaux de sessions, 30 min (ou 15) par palier, trois répétitions ;
   à comparer aux exports AgentX publiés.
   **Agents de test** (cache activé) : paliers jusqu'au-delà du palier retenu, 15 min, préchauffe 300 s,
   trois répétitions.
5. **À travers LiteLLM** : un balayage court (§ 7), pour la latence ajoutée.
6. **Reprise** : arrête une réplique, relance-la, et mesure le temps jusqu'à la première réponse
   (`curl -s -H "Authorization: Bearer $CLE_API" "$URL/v1/models"` toutes les 10 s) ; pas de balayage pendant
   ce test.
7. **Comparer** les configurations : débit synthétique avec et sans MTP, débit AgentX, débit des agents et
   taux de cache, TTFT P95, cache KV maximal, temps de reprise. Le choix de la configuration et de la cible ne
   relève pas du banc.
8. **En production** : observation continue (Prometheus en service, § 6.2) ; l'écart avec les stress tests montre ce qu'ils ne
   reproduisent pas de ta charge réelle.

Ordre de grandeur, par configuration et par contexte : ≈ 4 h pour les passes grossière et fine (5 paliers ×
15 min × 3), ≈ 7 h 30 pour AgentX (5 × 30 min × 3), ≈ 5 h 30 pour les agents de test avec 7 paliers (7 × 15 min × 3).

---

## 9. Interface (optionnel)

```bash
python interface/serveur.py            # option --port, défaut 8765
```

Ouvre **l'adresse exacte affichée** (elle contient un jeton d'accès). Sur une VM distante :
`ssh -L 8765:127.0.0.1:8765 <vm>`, puis la même adresse sur ton poste.

- **Lancer** : champs de connexion (mêmes sens qu'au § 2.2), bouton « Tester la connexion » qui affiche la
  configuration du serveur (version, cache de préfixes, TP/PP/DP/EP…), puis l'un des trois stress tests :
  synthétique, AgentX ou agents de test.
- **Essais** : journal en direct, arrêt.
- **Résultats** : tableaux, point retenu, courbe des stress tests.
- **Observation** : rien à lancer ; URL de Prometheus, plage et fenêtre, puis « Afficher » (vues par fenêtre,
  par période et par agent ; case « actualiser chaque minute »). Les plages des essais lancés depuis l'interface
  sont écartées (« essai en cours ») ; celles des essais lancés à la main, seulement si tu les donnes à
  `interroger.py --exclure`.

Les champs ne sont pas enregistrés, et l'interface n'hérite pas des variables de `cible.env`. Un seul essai
à la fois ; l'observation peut s'afficher pendant un essai. Détails : `interface/README.md`.

---

## 10. Ce que chaque essai enregistre

```
resultats/<étiquette>/config_serveur.json     # configuration du serveur au début de l'essai
resultats/<étiquette>/c<N>/                    # sorties AIPerf du palier de concurrence N (+ c<N>.log)
resultats-agentx/<étiquette>/c<N>/             # idem pour AgentX, plus agentx.txt (jeu de traces, durée)
agents_test/resultats-agents/<étiquette>/config_serveur.json     # lanceur lancé depuis agents_test/
agents_test/resultats-agents/<étiquette>/n<N>/   # appels.jsonl, taches.jsonl, palier.json (dont métriques vLLM)
```

L'observation n'écrit rien d'office : l'historique est dans Prometheus ; `interroger.py --sortie f.csv` écrit
les fenêtres d'une interrogation.

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

## 11. Dépannage

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
| AgentX : « Scenario invariants violated » | `DUREE` < 900 s, ou AIPerf lancé à la main sur des traces locales | `DUREE` ≥ 900 (ou `ESSAI_COURT=1` pour une simple vérification) ; passer par `agentx.sh` |
| AgentX : `err` > 0 au dépouillement | requêtes plus longues que `--max-model-len` (AIPerf les retire de ses métriques ; le dépouillement les compte dans `err`) | augmente-le ; `AGENTX_VARIANTE=256k` pour un modèle à ≈ 256 000 jetons ; ou `CONTEXTE_MAX` ≈ `--max-model-len` − 2 048 pour écarter les traces trop longues (§ 4.2) |
| AgentX : « Aucune trace ne tient » | `CONTEXTE_MAX` sous le contexte de toutes les traces | augmente `CONTEXTE_MAX` et `--max-model-len` ; sur un petit modèle, AgentX n'est pas faisable (§ 4.2) |
| AgentX : `timed out waiting for dataset configuration` | chargement des traces plus long que le délai d'AIPerf | `AIPERF_DATASET_CONFIGURATION_TIMEOUT` plus grand (§ 4) |
| `ERREUR : nombre de GPU inconnu` au dépouillement | empreinte absente, ou `METRIQUES` ne vise pas vLLM pendant l'essai | `--gpu N` (champ « GPU » de l'interface) |
| LiteLLM dans Docker : erreur de connexion vers vLLM | `127.0.0.1` désigne le conteneur, pas la machine | `api_base: http://host.docker.internal:<port de vLLM>/v1` ([Docker Desktop](https://docs.docker.com/desktop/features/networking/networking-how-tos/)) |
| `api_base` sur un port comme 56964 | port lu dans le journal de vLLM : c'est celui de l'appelant | le port de vLLM est son `--port` (8000 par défaut) |
| Observation : « part interpolée * » | pas de borne d'histogramme à la cible | `--custom-histogram-buckets` (§ 6.1) |
| Observation : « Aucune cible vLLM … dans Prometheus » | sélecteur faux, ou Prometheus ne collecte pas vLLM | page `http://<hôte>:9090/targets` : la cible doit être « UP » ; `--vllm 'job="<job_name>"'` |
| Observation : fenêtres « écartée » | cible injoignable, redémarrage de vLLM, ou collectes manquantes (message ATTENTION) | page Targets de Prometheus : erreur affichée ; pare-feu, adresse, vLLM arrêté |
| Observation : « Plage plus courte qu'une fenêtre » | `--depuis` ou `--debut`/`--fin` plus courts que `--fenetre` | allonge la plage ou raccourcis la fenêtre |
| Vue par agent vide | LiteLLM non collecté ou étiquette absente | cible `litellm` « UP » dans Prometheus ; `callbacks: ["prometheus"]`, `prometheus_client` installé, `metrics_path: /metrics/`, clé ; `--par` |
| Cible LiteLLM « DOWN » dans Prometheus, erreur 401 ou fichier de clé introuvable | clé absente ou fausse, ou `litellm.cle` manquant ou illisible | crée `litellm.cle` (§ 6.2) avec une clé valide ; `metrics_path: /metrics/` |
| Docker : « not a directory: Are you trying to mount a directory onto a file » | `prometheus.yml` ou `litellm.cle` absent au premier `up` : Docker a créé un dossier à sa place | `rmdir prometheus.yml litellm.cle`, crée les fichiers, relance (§ 6.2, B) |
| Observation : fenêtres « écartée » en série | `--collecte` plus court que le `scrape_interval` réel | donne la valeur de `prometheus.yml` |
| LiteLLM ne démarre pas avec `callbacks: ["prometheus"]` | module `prometheus_client` absent | `pip install prometheus_client` dans l'environnement de LiteLLM |

---

## 12. Vérifier la chaîne sans GPU

Un faux serveur compatible OpenAI permet de tester les scripts sans modèle (les chiffres ne mesurent rien).
Il imite aussi les histogrammes et jauges de vLLM et, sur `/litellm/metrics/`, les métriques de LiteLLM par clé
et par `metadata_agent`. L'observation se vérifie avec un Prometheus qui collecte le factice
(`outils/prometheus-factice.yml`, collecte toutes les 5 s, vLLM et « LiteLLM » sur le port 8001 ; le factice
n'expose pas `process_start_time_seconds`) :

```bash
python outils/serveur_factice.py --port 8001 --delai-jeton 0.02 \
  --bornes-itl '[0.01, 0.0133, 0.02, 0.0333, 0.05, 0.1]' &      # à arrêter ensuite : kill %1
cd agents_test && python lanceur.py --url http://127.0.0.1:8001 --paliers "1 2" --duree 20 \
  --prechauffe 2 --echelle 0.05 --facteur-latence 0.02 --etiquette essai --sortie /tmp/essai
python depouiller_agents.py /tmp/essai --cible 30          # GPU déduits de l'empreinte du factice : 8
# observation : Prometheus collecte le factice, une charge tourne 5 min, puis on interroge
cd .. && printf cle > outils/litellm.cle                      # ignoré par git
prometheus --config.file=outils/prometheus-factice.yml --storage.tsdb.path=/tmp/prom-data &   # à arrêter ensuite
(cd agents_test && python lanceur.py --url http://127.0.0.1:8001 --paliers "2" --duree 300 --prechauffe 2 \
  --echelle 0.05 --facteur-latence 0.05 --etiquette charge --sortie /tmp/charge --etiqueter-agents)
python observation/interroger.py --prometheus http://127.0.0.1:9090 --depuis 5m --fenetre 1m --collecte 5s \
  --cible 30 --gpu 8            # 8 : comme l'empreinte du factice
python observation/interroger.py --prometheus http://127.0.0.1:9090 --litellm 'job="litellm"' \
  --depuis 5m --fenetre 1m --collecte 5s --cible 30 --gpu 8 --vue agents
```

---

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `balayage.sh`, `depouiller.py` | stress test synthétique (AIPerf) et dépouillement des stress tests AIPerf |
| `agentx.sh`, `outils/telecharger_traces_agentx.sh` | stress test AgentX ; copie locale des traces |
| `observation/interroger.py`, `analyse.py` | mode observation : interrogation de Prometheus, verdicts (fenêtres, périodes, agents) |
| `outils/prometheus.yml.exemple`, `prometheus-factice.yml` | configuration de Prometheus (production, vérification sans GPU) |
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
- vLLM, métriques d'observation au commit `7f855f6` : [`loggers.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/v1/metrics/loggers.py) (dont [`vllm:num_requests_running`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/v1/metrics/loggers.py#L534)) ·
  [`stats.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/v1/metrics/stats.py) · [`buckets.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/v1/metrics/buckets.py) ·
  [`observability.py`](https://github.com/vllm-project/vllm/blob/7f855f605892656ebf7265642e5d22a34a6ae731/vllm/config/observability.py)
- AgentX : [méthode InferenceX](https://inferencex.semianalysis.com/blog/agentic-benchmark-agent-benchmark-guide) ·
  [jeu de traces `cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) ·
  scénario AIPerf : `aiperf/common/scenario/inferencex_agentx_mvp.py` dans le paquet installé
- Prometheus : [présentation](https://prometheus.io/docs/introduction/overview/) ·
  [configuration (`scrape_config`, `authorization`)](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#scrape_config) ·
  [fonctions PromQL (`resets`)](https://prometheus.io/docs/prometheus/latest/querying/functions/#resets) ·
  [API HTTP (`query_range`)](https://prometheus.io/docs/prometheus/latest/querying/api/#range-queries) ·
  [opérateurs (`and`, `unless`)](https://prometheus.io/docs/prometheus/latest/querying/operators/#logical-set-binary-operators) ·
  [`offset`](https://prometheus.io/docs/prometheus/latest/querying/basics/#offset-modifier) ·
  [téléchargement](https://prometheus.io/download/) ; testé avec Prometheus 3.7.3
- [AIPerf 0.12.0](https://pypi.org/project/aiperf/0.12.0/) ·
  [balayage de concurrence (NVIDIA)](https://docs.nvidia.com/nim/benchmarking/llm/latest/step-by-step.html)
- LiteLLM : [fournisseur vLLM](https://docs.litellm.ai/docs/providers/vllm) · [configuration du proxy](https://docs.litellm.ai/docs/proxy/configs) ·
  [métriques Prometheus](https://docs.litellm.ai/docs/proxy/prometheus)
