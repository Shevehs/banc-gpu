# Interface du banc

Petite application locale pour lancer les stress tests du banc, suivre leur journal, afficher les résultats et
interroger Prometheus pour l'observation de l'usage réel. Elle appelle les mêmes scripts que la ligne de
commande (`balayage.sh`, `agentx.sh`, `agents_test/lanceur.py`, `observation/interroger.py` et les dépouilleurs) :
un essai lancé ici est identique au même essai lancé à la main, et la commande exacte est écrite en tête de
son journal.

## Lancer

Depuis la racine du dépôt, environnement `.venv-aiperf` activé (il contient déjà `aiohttp`) :

```bash
python interface/serveur.py
```

puis ouvrir **l'adresse exacte affichée** (elle contient un jeton d'accès : `http://127.0.0.1:8765/?jeton=…`).

Sur une machine distante (VM cliente), le serveur n'écoute que sur 127.0.0.1 ; depuis ton poste :

```bash
ssh -L 8765:127.0.0.1:8765 <nœud>
```

puis ouvrir sur ton poste l'adresse affichée par le serveur (avec son jeton).

L'interface ne démarre jamais vLLM : elle mesure celui qui tourne. Options à vérifier selon le mode : § 1.2 du README du dépôt.

## Sécurité

- Écoute sur 127.0.0.1 seulement ; jeton aléatoire exigé pour toute l'API (affiché au démarrage, ou fixé par
  la variable `BANC_JETON`) ; contrôle de l'en-tête Host et de l'origine des requêtes.
- N'exécute que les scripts du banc, avec des paramètres vérifiés ; options AIPerf supplémentaires sur liste
  blanche. Un seul essai à la fois ; arrêter l'interface arrête l'essai en cours.
- La clé d'API et les clés par agent passent par variables d'environnement et ne sont écrites dans aucun journal.
- Les essais n'héritent pas des réglages du terminal (`URL`, `CLE_API`, `EXTRA`…) : seuls les champs de
  l'interface comptent ; du terminal, seuls passent `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `TMPDIR`,
  la langue (`LANG`, `LC_*`), le fuseau horaire (`TZ`), `VIRTUAL_ENV`, `HF_HOME`, `HF_HUB_CACHE`, le proxy, les
  certificats et `DO_NOT_TRACK`.
- La clé d'API va à la cible ; aucune clé ne va au `/metrics` de vLLM. En observation, seule la « Clé de
  Prometheus » (si remplie) part, vers Prometheus ; la clé de LiteLLM est dans la configuration de Prometheus,
  pas dans l'interface.

## Ce qu'elle fait

- **Lancer** : champs de connexion (cible vLLM ou LiteLLM, modèle, métriques vLLM, clé d'API), bouton « Tester la connexion » (qui affiche aussi la configuration du serveur vLLM : version, cache de préfixes, nombre de jetons spéculatifs, parallélisme TP/PP/DP/EP lu dans l'empreinte des réponses et, seulement si vLLM est lancé avec `VLLM_SERVER_DEV_MODE=1`, méthode de décodage spéculatif et ordonnanceur), puis l'un des trois stress tests : synthétique
  (AIPerf), AgentX ou agents de test (README du dépôt, § 3 à 5).
  Pour les agents de test, « Étiqueter les agents » et « Clés par agent » donnent à chaque requête l'identité
  de son profil, pour la vue par agent via LiteLLM (README du dépôt, § 6.5).
- **Essais** : liste des essais lancés depuis le démarrage du serveur de l'interface, journal en direct, arrêt.
  Les journaux restent dans `interface/journaux/`.
- **Résultats** : dépouillement d'un dossier de résultats (tableau, point retenu, avertissements, courbe
  débit par GPU selon l'interactivité, avec la cible).
- **Observation** : rien à lancer. URL de Prometheus, sélecteurs des cibles vLLM et LiteLLM, plage (« Depuis »,
  ou début et fin), durée des fenêtres, puis « Afficher » : vues par fenêtre, par période (plages de 6 h ou
  heures) et par agent (README du dépôt, § 6). Case « actualiser chaque minute » pour suivre l'usage en direct.
  Les plages des essais lancés depuis l'interface (depuis le démarrage de son serveur) sont marquées « essai en cours » et
  écartées des verdicts. Fonctionne pendant qu'un essai tourne.

## Modifier l'interface

Les fichiers compilés sont dans `interface/dist/` et versionnés : aucune installation de Node n'est
nécessaire pour s'en servir. Pour modifier l'interface (Node 18 ou plus) :

```bash
cd interface/web && npm install && npm run build
```
