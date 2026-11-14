"""Lance les agents de test par paliers de tâches simultanées et journalise chaque appel.

Exemple (petit modèle, contexte de 32 768 jetons) :
  python lanceur.py --url http://127.0.0.1:8000 --modele banc --agents revue_code,pentest,rapport \
      --paliers "1 2 4" --duree 600 --prechauffe 120 --echelle 0.2 --etiquette essai-r1

--url vise vLLM directement ou une passerelle (LiteLLM). Les métriques du serveur (préemptions,
cache, décodage spéculatif, cache KV) sont lues sur --metriques : par défaut <url>/metrics, ce qui
convient quand --url est vLLM ; avec une passerelle, donner les URL /metrics de vLLM (une par réplique).
Clé d'API : variable d'environnement BANC_CLE_API ou CLE_API (jamais en argument, pour ne pas apparaître dans les
journaux ni la liste des processus).

Pour chaque palier N, N « travailleurs » enchaînent des tâches (agents pris à tour de rôle)
pendant --duree secondes ; aucune nouvelle tâche ne démarre ensuite, et les tâches en cours
se terminent. Résultats : <sortie>/<etiquette>/n<N>/{appels,taches}.jsonl et palier.json.
"""
import argparse, itertools, json, os, random, re, sys, threading, time, traceback, zlib

import agents as module_agents
from agents import AGENTS
from commun import Journal, Modele
from metriques import Releve


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--url", default="http://127.0.0.1:8000")
    a.add_argument("--modele", default="banc", help="nom servi (--served-model-name)")
    a.add_argument("--agents", default="revue_code,pentest,rapport")
    a.add_argument("--paliers", default="1 2 4 8", help="tâches simultanées, séparées par des espaces")
    a.add_argument("--duree", type=float, default=900, help="secondes par palier")
    a.add_argument("--prechauffe", type=float, default=300,
                   help="secondes exclues du dépouillement en début de palier (au moins la durée d'une tâche)")
    a.add_argument("--etalement", type=float, default=30, help="départs des travailleurs étalés sur ce nombre de secondes")
    a.add_argument("--echelle", type=float, default=1.0, help="multiplie toutes les tailles (0,2 pour un modèle à 32 768 jetons de contexte)")
    a.add_argument("--mode", choices=["scripte", "autonome"], default="scripte")
    a.add_argument("--facteur-latence", type=float, default=1.0, help="multiplie les temps d'outil simulés")
    a.add_argument("--metriques", default=None,
                   help="URL /metrics de vLLM, séparées par des espaces (défaut : <url>/metrics)")
    a.add_argument("--sans-metriques", action="store_true", help="ne pas lire /metrics")
    a.add_argument("--sortie", default="resultats-agents")
    a.add_argument("--etiquette", required=True)
    a.add_argument("--ecraser", action="store_true", help="remplace des résultats existants de même étiquette")
    p = a.parse_args()
    module_agents.FACTEUR_LATENCE = p.facteur_latence
    cle = os.environ.get("BANC_CLE_API") or os.environ.get("CLE_API") or None
    if p.prechauffe >= p.duree:
        raise SystemExit(f"--prechauffe ({p.prechauffe:.0f} s) doit être inférieure à --duree ({p.duree:.0f} s) : "
                         "sinon la fenêtre de mesure est vide.")
    urls_metriques = [] if p.sans_metriques else (p.metriques.split() if p.metriques else [p.url.rstrip("/") + "/metrics"])
    agents = [x.strip() for x in p.agents.split(",") if x.strip()]
    inconnus = [x for x in agents if x not in AGENTS]
    if inconnus:
        raise SystemExit(f"Agents inconnus : {inconnus} (choix : {list(AGENTS)})")

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$", p.etiquette) or ".." in p.etiquette:
        raise SystemExit("--etiquette : lettres, chiffres, point, tiret, souligné, en commençant par une lettre "
                         "ou un chiffre ; ni / ni ..")
    deja = os.path.join(p.sortie, p.etiquette)
    if os.path.isdir(deja) and os.listdir(deja):
        if not p.ecraser:
            raise SystemExit(f"{deja} existe déjà : change d'étiquette, ou ajoute --ecraser pour le remplacer.")
        import shutil; shutil.rmtree(deja)
    os.makedirs(deja, exist_ok=True)
    try:  # instantané de la configuration du serveur (version, cache, parallélisme par l'empreinte)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils"))
        from config_serveur import releve as releve_config, resume
        cfg = releve_config(p.url, urls_metriques, cle, p.modele)
        with open(os.path.join(deja, "config_serveur.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1, ensure_ascii=False)
        print(resume(cfg), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(configuration du serveur non relevée : {e})", file=sys.stderr)
    for n in [int(x) for x in p.paliers.split()]:
        dossier = os.path.join(p.sortie, p.etiquette, f"n{n}")
        os.makedirs(dossier, exist_ok=True)
        journal = Journal(dossier)
        t_debut = time.time(); t_fin = t_debut + p.duree
        compteur = itertools.count()
        arret = threading.Event()
        releve = Releve(urls_metriques, t_debut + p.prechauffe, t_fin).demarrer()  # /metrics : sans clé
        print(f"== {p.etiquette} : {n} tâche(s) simultanée(s), {p.duree:.0f} s ==", flush=True)

        def travailleur(w):
            m = Modele(p.url, p.modele, journal, p.mode, cle_api=cle or "banc")
            k = w  # décalage : les travailleurs ne commencent pas tous par le même agent
            # départs étalés, pour éviter des vagues synchronisées
            arret.wait(random.Random(f"{p.etiquette}-{n}-{w}").uniform(0, min(p.etalement, p.duree / 4)))
            while time.time() < t_fin and not arret.is_set():
                agent = agents[k % len(agents)]; k += 1
                num = next(compteur); tache = f"{p.etiquette}-n{n}-{num}"
                t0 = time.time(); err = None
                try:
                    AGENTS[agent](m, tache, graine=zlib.crc32(f"{p.etiquette}-n{n}-{num}".encode()), echelle=p.echelle, mode=p.mode)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:300]
                    traceback.print_exc()
                journal.tache(agent=agent, tache=tache, travailleur=w, t_debut=t0, t_fin=time.time(), erreur=err)
                if err:
                    time.sleep(2)  # évite une boucle d'erreurs rapide si le serveur est tombé

        fils = [threading.Thread(target=travailleur, args=(w,), daemon=True) for w in range(n)]
        for f in fils: f.start()
        try:
            while any(f.is_alive() for f in fils):
                time.sleep(0.5)
        except KeyboardInterrupt:
            arret.set(); journal.fermer()
            print("\nInterrompu : les tâches en cours sont abandonnées ; ce palier est incomplet.", flush=True)
            sys.exit(130)
        journal.fermer()
        mesures = releve.resultat()
        if urls_metriques and mesures["lectures"] < 2:
            print(f"   ATTENTION : métriques du serveur non lues ({mesures['lectures_echouees']} échec(s)) sur {urls_metriques}", flush=True)
        with open(os.path.join(dossier, "palier.json"), "w") as fh:
            json.dump({"taches_simultanees": n, "t_debut": t_debut, "t_fin": t_fin, "t_fin_reelle": time.time(),
                       "prechauffe": p.prechauffe, "echelle": p.echelle, "facteur_latence": p.facteur_latence, "mode": p.mode, "agents": agents,
                       "url": p.url, "modele": p.modele, "metriques": mesures}, fh, indent=1)
        print(f"   terminé en {time.time() - t_debut:.0f} s -> {dossier}", flush=True)


if __name__ == "__main__":
    main()
