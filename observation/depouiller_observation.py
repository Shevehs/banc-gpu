"""Dépouille les fenêtres d'observation : une ligne par fenêtre, puis, si des niveaux ont été notés,
le plus haut niveau qui tient la cible.
Usage : python observation/depouiller_observation.py resultats-observation --cible 30 --gpu 8 [--ttft-max ms]
        [--vue fenetres|periode|agents] [--periode heure|plage]
  --cible : interactivité visée par requête (jetons/s). Une fenêtre « tient » si au moins 90 % des requêtes
            terminées l'atteignent (même seuil que l'interactivité P10 des stress tests, mesuré ici par vLLM),
            sans préemption ni requête en erreur (et avec un TTFT P95 ≤ --ttft-max si donné). « Indéterminé » :
            la cible tombe dans un seau d'histogramme qui encadre 90 % (bornes à ajouter, README § 6.1).
  --gpu   : nombre de GPU derrière les URL de métriques, toutes répliques comprises
  --vue periode : par configuration, regroupe les fenêtres par heure (--periode heure) ou par plage de 6 h
                  (nuit 0-6 h, matin 6-12 h, après-midi 12-18 h, soir 18-24 h ; --periode plage), heure locale
  --vue agents  : jetons, requêtes, TTFT et latence par agent, lus dans les métriques de LiteLLM
                  (observer.py --litellm) ; toutes les fenêtres d'une configuration additionnées
Regroupement : étiquette sans ses suffixes -n<niveau> et -r<répétition> (ex. prod-n4-r1 -> prod).
Écrit <racine>/tableau_observation.csv. Bibliothèque standard seulement."""
import argparse, csv, json, os, pathlib, re, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse import indicateurs, indicateurs_agents, verdict  # noqa: E402

PLAGES = [(0, "nuit (0-6 h)"), (6, "matin (6-12 h)"), (12, "après-midi (12-18 h)"), (18, "soir (18-24 h)")]
f = lambda x, n=1: "—" if x is None or x == "" else f"{float(x):.{n}f}"
pc = lambda x: "—" if x is None or x == "" else f"{100 * float(x):.0f}%"


def periode(t, mode):
    h = time.localtime(t).tm_hour
    return f"{h:02d} h" if mode == "heure" else [n for d, n in PLAGES if h >= d][-1]


def somme_histo(hs):
    r = {}
    for h in hs:
        for le, n in h.items():
            r[le] = r.get(le, 0.0) + n
    return r


def fusionner(obs):
    """Additionne des fenêtres successives (compteurs, histogrammes ; jauges : maximum) en une pseudo-fenêtre."""
    c = {}
    for o in obs:
        for k, v in o["compteurs"].items():
            if k == "requetes_en_erreur":  # None = aucune série /v1 encore créée : rien à ajouter
                c[k] = c.get(k) if v is None else (c.get(k) or 0.0) + v
            else:  # autre compteur illisible dans une fenêtre : somme inconnue
                c[k] = None if v is None or (k in c and c[k] is None) else c.get(k, 0.0) + v
    return {"debut": 0.0, "fin": sum(o["fin"] - o["debut"] for o in obs), "compteurs": c,
            "histogrammes": {k: somme_histo([o["histogrammes"][k] for o in obs]) for k in obs[0]["histogrammes"]},
            "jauges": {k: max((o["jauges"][k] for o in obs if o["jauges"][k] is not None), default=None)
                       for k in obs[0]["jauges"]}}


def chevauchements(obs):
    """Vrai si deux fenêtres se recouvrent dans le temps (même serveur observé deux fois, ou autre serveur
    sous la même étiquette) : leur somme compterait des requêtes deux fois, ou mélangerait des serveurs."""
    o = sorted(obs, key=lambda x: x["debut"])
    return any(b["debut"] < a["fin"] - 1 for a, b in zip(o, o[1:]))


def part_txt(r):
    if r["part_requetes_a_la_cible"] is None:
        return "—"
    return pc(r["part_requetes_a_la_cible"]) + ("" if r["part_exacte"] else "*")


def entete():
    return (f"{'requêtes':>9}{'err':>5}{'débit/GPU':>11}{'à la cible':>11}{'inter P10≥':>11}"
            f"{'TTFT P95≤ ms':>13}{'préempt':>8}{'cache':>7}{'MTP':>6}{'KV max':>8}  verdict")


def ligne(r):
    return (f"{f(r['requetes_terminees'], 0):>9}{f(r['requetes_en_erreur'], 0):>5}{f(r['debit_sortie_par_gpu']):>11}"
            f"{part_txt(r):>11}{('<' if r['interactivite_borne'] else '') + f(r['interactivite_p10_estimee']):>11}"
            f"{('>' if r['ttft_p95_au_dela'] else '') + f(r['ttft_p95_ms_estime'], 0):>13}{f(r['preemptions'], 0):>8}"
            f"{pc(r['cache_prefixes']):>7}{pc(r['acceptation_mtp']):>6}{pc(r['cache_kv_max_fraction']):>8}"
            f"  {r.get('verdict') or verdict(r['tient'])}")


def vue_periode(valides, p):
    groupes = {}
    for o, conf in valides:
        groupes.setdefault((conf, periode(o["debut"], p.periode)), []).append(o)
    print(f"{'configuration':<16}{'période':<22}{'fenêtres':>9}" + entete())
    for (conf, per) in sorted(groupes):
        obs = groupes[(conf, per)]
        r = indicateurs(fusionner(obs), p.cible, p.gpu, p.ttft_max)
        melange = chevauchements(obs) or len({tuple(sorted(o.get("urls") or [])) for o in obs}) > 1
        print(f"{conf[:16]:<16}{per:<22}{len(obs):>9}" + (ligne(r) if not melange else
              ligne({**r, "tient": None}).replace("indéterminé", "mélange")))
        if melange:
            print(f"   ATTENTION : {conf}, {per} : fenêtres qui se recouvrent ou serveurs différents sous la même étiquette : "
                  "pas de verdict (somme faussée) ; donne une étiquette par serveur ou par ensemble de répliques.")
    print("\nDébit/GPU : moyenne sur la période (jetons de sortie ÷ durée observée ÷ GPU). Pointe d'une période : "
          "regarde ses fenêtres une par une (--vue fenetres).")


def vue_agents(valides):
    confs = {}
    for o, conf in valides:
        if o.get("agents") is not None:
            confs.setdefault(conf, []).append(o)
    if not confs:
        print("Aucune donnée par agent : lance observer.py avec --litellm (callback prometheus de LiteLLM).")
        return
    for conf, obs in sorted(confs.items()):
        tous = {}
        for o in obs:
            for ag, a in o["agents"].items():
                t = tous.setdefault(ag, {k: ({} if isinstance(v, dict) else 0.0) for k, v in a.items()})
                for k, v in a.items():
                    t[k] = somme_histo([t[k], v]) if isinstance(v, dict) else t[k] + v
        par = obs[0].get("agents_par", "api_key_alias")
        print(f"{conf} ({len(obs)} fenêtre(s), agents distingués par {par})")
        print(f"  {'agent':<24}{'requêtes':>9}{'part sortie':>12}{'entrée/req':>11}{'sortie/req':>11}"
              f"{'TTFT P50≤ s':>12}{'TTFT P95≤ s':>12}{'latence P95≤ s':>15}{'vitesse moy':>12}")
        for ag, r in indicateurs_agents(tous).items():
            print(f"  {ag[:24]:<24}{f(r['requetes'], 0):>9}{pc(r['part_jetons_sortie']):>12}"
                  f"{f(r['jetons_entree_par_requete'], 0):>11}{f(r['jetons_sortie_par_requete'], 0):>11}"
                  f"{f(r['ttft_p50_s'], 2):>12}{('>' if r['ttft_p95_au_dela'] else '') + f(r['ttft_p95_s'], 2):>12}"
                  f"{('>' if r['latence_p95_au_dela'] else '') + f(r['latence_p95_s'], 2):>15}{f(r['vitesse_moyenne']):>12}")
        if "None" in tous or "" in tous:
            print("   « None » ou vide : requêtes sans identité d'agent (sans clé virtuelle, ou sans metadata.agent).")
    print("\nVue par agent lue côté LiteLLM : TTFT seulement pour les requêtes en streaming ; vitesse moyenne = jetons "
          "de sortie ÷ (latence − TTFT), moyenne et non centile, temps du proxy compris, vide si une partie des "
          "requêtes n'est pas en streaming. Bornes ≤ : seaux des histogrammes de LiteLLM.")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("racine"); a.add_argument("--cible", type=float, required=True)
    a.add_argument("--gpu", type=int, required=True)
    a.add_argument("--ttft-max", type=float, default=None, help="TTFT P95 estimé maximal accepté (ms)")
    a.add_argument("--vue", choices=["fenetres", "periode", "agents"], default="fenetres")
    a.add_argument("--periode", choices=["heure", "plage"], default="plage")
    p = a.parse_args()
    lignes, alertes, valides = [], [], []
    for fichier in sorted(pathlib.Path(p.racine).glob("*/observation*.json")):
        o = json.loads(fichier.read_text())
        etiq = o["etiquette"]; conf = re.sub(r"(-n\d+)?(-r\d+)?$", "", etiq)
        debut = time.strftime("%Y-%m-%d %H:%M", time.localtime(o["debut"]))
        if o.get("incomplete"):
            alertes.append(f"{etiq} {debut} : fenêtre incomplète ({o.get('erreur')}) : écartée.")
            continue
        r = indicateurs(o, p.cible, p.gpu, p.ttft_max)
        ecartee = bool(o.get("remise_a_zero"))
        if ecartee:
            alertes.append(f"{etiq} {debut} : serveur redémarré pendant la fenêtre : fenêtre écartée.")
            r = {k: (None if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in r.items()}
            r["tient"] = False
        else:
            valides.append((o, conf))
        if o.get("lectures_echouees"):
            alertes.append(f"{etiq} {debut} : {o['lectures_echouees']} lecture(s) de /metrics échouée(s) : jauges incomplètes.")
        if not ecartee and (r["requetes_terminees"] or 0) < 50:
            alertes.append(f"{etiq} {debut} : moins de 50 requêtes terminées : centiles peu fiables.")
        if o.get("agents_erreur"):
            alertes.append(f"{etiq} {debut} : métriques LiteLLM non lues ({o['agents_erreur']}).")
        lignes.append({"configuration": conf, "fenetre": etiq, "debut": debut, "niveau": o.get("niveau"),
                       "duree_s": round(o["fin"] - o["debut"]), **r,
                       "verdict": "écartée" if ecartee else verdict(r["tient"]), "ecartee": ecartee})
    if not lignes:
        for al in alertes:
            print("   ATTENTION : " + al)
        raise SystemExit(f"Aucune fenêtre exploitable sous {p.racine}/<étiquette>/")
    with open(pathlib.Path(p.racine) / "tableau_observation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0])); w.writeheader(); w.writerows(lignes)
    if p.vue == "periode":
        vue_periode(valides, p)
    elif p.vue == "agents":
        vue_agents(valides)
    else:
        print(f"Tableau : {pathlib.Path(p.racine) / 'tableau_observation.csv'}\n")
        print(f"{'fenêtre':<20}{'début':>17}{'niveau':>7}{'durée s':>8}" + entete())
        for l in lignes:
            print(f"{l['fenetre'][:20]:<20}{l['debut']:>17}{f(l['niveau'], 0):>7}{l['duree_s']:>8}" + ligne(l))
        if any(l["part_exacte"] is False for l in lignes):
            print(f"\n* part interpolée : aucune borne d'histogramme à {1000 / p.cible:.1f} ms ; lance vLLM avec "
                  "--custom-histogram-buckets (README, § 6.1). « indéterminé » : la cible tombe dans un seau qui encadre 90 %.")
        print("≥ et ≤ : bornes lues dans les seaux des histogrammes (< ou > : au-delà de la dernière borne). "
              "err : requêtes /v1 en 4xx ou 5xx vues par vLLM (« — » si non exposé).")
        # Plus haut niveau qui tient, par configuration : médiane des répétitions pour la part à la cible et le
        # TTFT ; aucune préemption ni erreur ; aucune fenêtre écartée.
        print()
        groupes = {}
        for l in lignes:
            if l["niveau"] is not None:
                groupes.setdefault((l["configuration"], l["niveau"]), []).append(l)
        for conf in sorted({k[0] for k in groupes}):
            niveaux = sorted(n for (c, n) in groupes if c == conf)
            ok = []
            for n in niveaux:
                ls = groupes[(conf, n)]
                mins = [l["part_min"] for l in ls if l["part_min"] is not None]
                t95 = [l["ttft_p95_ms_estime"] for l in ls if l["ttft_p95_ms_estime"] is not None]
                if (mins and statistics.median(mins) >= 0.90 and not any(l["ecartee"] for l in ls)
                        and sum(l["preemptions"] or 0 for l in ls) == 0 and sum(l["requetes_en_erreur"] or 0 for l in ls) == 0
                        and (p.ttft_max is None or (t95 and statistics.median(t95) <= p.ttft_max))):
                    ok.append((n, statistics.median([l["debit_sortie_par_gpu"] or 0 for l in ls]), len(ls)))
            if not ok:
                print(f"{conf} : aucun niveau ne tient {p.cible:g} jetons/s pour 90 % des requêtes sans préemption "
                      "ni erreur (ou résultat indéterminé : voir les verdicts).")
                continue
            n, deb, k = max(ok)
            print(f"{conf} : plus haut niveau qui tient = {n} ({deb:.1f} jetons/s par GPU, {k} fenêtre(s)).")
            if n == niveaux[-1]:
                print("   ATTENTION : c'est le plus haut niveau observé ; la limite n'est pas atteinte.")
            lr = groupes[(conf, n)]
            if any((l["requetes_en_attente_max"] or 0) > 0 for l in lr):
                print("   ATTENTION : des requêtes ont attendu dans la file du serveur à ce niveau.")
            if any((l["cache_kv_max_fraction"] or 0) >= 0.95 for l in lr):
                print("   ATTENTION : cache KV rempli à 95 % ou plus à ce niveau.")
    for al in alertes:
        print("   ATTENTION : " + al)


if __name__ == "__main__":
    main()
