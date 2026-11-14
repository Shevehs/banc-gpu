"""Dépouille les paliers du lanceur d'agents : une ligne par palier, plus le détail par agent.
Usage : python depouiller_agents.py resultats-agents --cible 30 [--gpu 8] [--exiger-metriques]
  --gpu : facultatif ; sans lui, déduit de config_serveur.json de chaque essai (TP × PP × DP × répliques)
Les métriques du serveur (préemptions, cache de préfixes côté serveur, acceptation MTP, cache KV maximal)
viennent du relevé /metrics fait par le lanceur (palier.json) ; elles figurent sur la ligne « tous ».
Fenêtre retenue : du début du palier + préchauffe jusqu'à la fin prévue du palier ; seuls les appels
terminés dans cette fenêtre comptent (la queue, où la concurrence baisse, est exclue).
Bibliothèque standard seulement. Écrit <racine>/tableau_agents.csv."""
import argparse, csv, json, pathlib, re, statistics


def centile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q; i = int(k); f = k - i
    return xs[i] if i + 1 >= len(xs) else xs[i] * (1 - f) + xs[i + 1] * f


def lire(f):
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()] if f.exists() else []


def nombre_gpu(dossier_essai, gpu_donne, notes):
    """GPU derrière la cible pour un essai : --gpu s'il est donné, sinon déduit de son config_serveur.json
    (TP × PP × DP de l'empreinte × répliques). Note l'origine dans `notes` ; None si rien ne permet de le savoir."""
    import sys as _s
    _s.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "outils"))
    from config_serveur import gpu_essai
    n, expl, masque = gpu_essai(dossier_essai)
    nom = pathlib.Path(dossier_essai).name
    if gpu_donne:
        if n and n != gpu_donne and not masque:
            notes[nom] = (f"ATTENTION : {nom} : --gpu {gpu_donne}, mais l'empreinte donne {expl} : vérifie la valeur.")
        return gpu_donne
    if n is None:
        notes[nom] = expl
        return None
    notes[nom] = (f"GPU : {nom} : {expl}, déduits de l'empreinte"
                  + (" ; ATTENTION : 1 GPU par serveur supposé, car l'empreinte n'a pas de degré de parallélisme "
                     "(--fingerprint-mode hash le masquerait) : donne --gpu si le serveur en a plus." if masque else "."))
    return n


def bilan(appels, t0, t1, n_gpu):
    fen = [x for x in appels if t0 <= x["t_debut"] + x["duree_s"] <= t1]
    ok = [x for x in fen if not x.get("erreur")]
    sortie = sum(x["jetons_sortie"] or 0 for x in ok)
    ent = [x for x in ok if x["jetons_entree"] and x["jetons_cache"] is not None]
    vit = [x["vitesse"] for x in ok if (x["jetons_sortie"] or 0) >= 16]
    return {
        "appels": len(fen), "erreurs_appels": len(fen) - len(ok),
        "debit_sortie_par_gpu": sortie / (t1 - t0) / n_gpu if t1 > t0 else None,
        "interactivite_p10": centile(vit, 0.10), "interactivite_p50": centile(vit, 0.50),
        "ttft_p50_s": centile([x["ttft_s"] for x in ok], 0.50), "ttft_p95_s": centile([x["ttft_s"] for x in ok], 0.95),
        "taux_cache": (sum(x["jetons_cache"] for x in ent) / sum(x["jetons_entree"] for x in ent)) if ent else None,
        "appels_outils": sum(x.get("appels_outils") or 0 for x in ok),
        "contexte_max": max((x["jetons_entree"] or 0) + (x["jetons_sortie"] or 0) for x in ok) if ok else None,
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument("racine"); a.add_argument("--cible", type=float, required=True); a.add_argument("--gpu", type=int, default=None)
    a.add_argument("--exiger-metriques", action="store_true", help="refuse un palier sans métriques du serveur")
    p = a.parse_args()
    if p.gpu is not None and p.gpu < 1:
        raise SystemExit("--gpu : entier ≥ 1")
    lignes, notes_gpu, sans_gpu = [], {}, {}
    for fp in sorted(pathlib.Path(p.racine).glob("*/n*/palier.json"), key=lambda f: (f.parent.parent.name, int(f.parent.name[1:]))):
        d = fp.parent; meta = json.loads(fp.read_text())
        n_gpu = nombre_gpu(d.parent, p.gpu, notes_gpu)
        if n_gpu is None:
            sans_gpu[d.parent.name] = notes_gpu.pop(d.parent.name, "?"); continue
        t0 = meta["t_debut"] + meta["prechauffe"]; t1 = meta["t_fin"]
        appels = lire(d / "appels.jsonl"); taches = lire(d / "taches.jsonl")
        tf = [t for t in taches if not t.get("erreur")]
        b = bilan(appels, t0, t1, n_gpu); b["gpu"] = n_gpu
        ms = meta.get("metriques") or {}
        rapport = lambda a, b_: (a / b_) if a is not None and b_ else None
        serveur = {"preemptions": ms.get("preemptions"),
                   "cache_prefixes_serveur": rapport(ms.get("cache_prefixes_reussites"), ms.get("cache_prefixes_requetes")),
                   "acceptation_mtp": rapport(ms.get("mtp_acceptes"), ms.get("mtp_proposes")),
                   "cache_kv_max_fraction": ms.get("cache_kv_max_fraction"),
                   "requetes_en_attente_max": ms.get("requetes_en_attente_max")}
        lignes.append({"essai": d.parent.name, "taches_simultanees": meta["taches_simultanees"], "agent": "tous",
                       "taches_finies": len(tf), "taches_en_erreur": len(taches) - len(tf),
                       "duree_tache_p50_min": (centile([t["t_fin"] - t["t_debut"] for t in tf], .5) or 0) / 60, **b, **serveur})
        for ag in meta["agents"]:
            ba = bilan([x for x in appels if x["agent"] == ag], t0, t1, n_gpu)
            ta = [t for t in tf if t["agent"] == ag]
            lignes.append({"essai": d.parent.name, "taches_simultanees": meta["taches_simultanees"], "agent": ag,
                           "taches_finies": len(ta), "taches_en_erreur": sum(1 for t in taches if t["agent"] == ag and t.get("erreur")),
                           "duree_tache_p50_min": (centile([t["t_fin"] - t["t_debut"] for t in ta], .5) or 0) / 60, **ba,
                           "preemptions": None, "cache_prefixes_serveur": None, "acceptation_mtp": None, "cache_kv_max_fraction": None,
                           "requetes_en_attente_max": None})
    if sans_gpu:
        raise SystemExit("ERREUR : nombre de GPU inconnu pour "
                         + " ; ".join(f"{e} ({r})" for e, r in sorted(sans_gpu.items()))
                         + " : donne --gpu (champ « GPU » dans l'interface).")
    for n in notes_gpu.values():
        print(n)
    par_conf = {}
    for l in lignes:
        if l.get("gpu") is not None:
            par_conf.setdefault(re.sub(r"-r\d+$", "", l["essai"]), {})[l["essai"]] = l["gpu"]
    for conf, ess in sorted(par_conf.items()):
        if len(set(ess.values())) > 1:
            print(f"ATTENTION : {conf} : nombre de GPU différent selon les répétitions ("
                  + ", ".join(f"{e} = {g}" for e, g in sorted(ess.items()))
                  + ") : la médiane mélange des configurations ; vérifie, ou donne --gpu.")
    if not lignes:
        raise SystemExit(f"Aucun palier sous {p.racine}/<essai>/n<N>/")
    with open(pathlib.Path(p.racine) / "tableau_agents.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0])); w.writeheader(); w.writerows(lignes)
    f = lambda x, n=1: "—" if x is None else f"{x:.{n}f}"
    pc = lambda x: "—" if x is None else f"{100 * x:.0f}%"
    print(f"{'essai':<12}{'N':>3} {'agent':<11}{'tâches':>7}{'err':>5}{'débit/GPU':>11}{'inter P10':>10}{'TTFT P95 s':>11}"
          f"{'cache':>7}{'durée min':>10}{'préempt':>9}{'MTP':>6}{'KV max':>8}")
    for l in lignes:
        print(f"{l['essai']:<12}{l['taches_simultanees']:>3} {l['agent']:<11}{l['taches_finies']:>7}{l['taches_en_erreur'] + l['erreurs_appels']:>5}"
              f"{f(l['debit_sortie_par_gpu']):>11}{f(l['interactivite_p10']):>10}{f(l['ttft_p95_s'], 2):>11}"
              f"{pc(l['taux_cache']):>7}{f(l['duree_tache_p50_min']):>10}{f(l['preemptions'], 0):>9}"
              f"{pc(l['acceptation_mtp']):>6}{pc(l['cache_kv_max_fraction']):>8}")
    # Palier retenu : le plus grand nombre de tâches simultanées dont l'interactivité P10 (tous agents)
    # atteint la cible, sans erreur ni préemption — médiane des répétitions (essais <conf>-rN).
    groupes = {}
    for l in lignes:
        if l["agent"] == "tous":
            groupes.setdefault((re.sub(r"-r\d+$", "", l["essai"]), l["taches_simultanees"]), []).append(l)
    print()
    for conf in sorted({k[0] for k in groupes}):
        paliers = sorted(n for (c, n) in groupes if c == conf)
        ok, refus, sans = [], [], False
        for n in paliers:
            ls = groupes[(conf, n)]
            manque = any(x["preemptions"] is None for x in ls); sans |= manque
            if (statistics.median([x["interactivite_p10"] or 0 for x in ls]) >= p.cible
                    and sum(x["taches_en_erreur"] + x["erreurs_appels"] for x in ls) == 0
                    and sum(x["preemptions"] or 0 for x in ls) == 0
                    and not (manque and p.exiger_metriques)):
                debs = [x["debit_sortie_par_gpu"] or 0 for x in ls]
                ok.append((n, statistics.median(debs), len(ls), min(debs), max(debs)))
            else:
                refus.append(n)
        if not ok:
            print(f"{conf} : aucun palier n'atteint {p.cible} jetons/s par fil (P10) sans erreur ni préemption.")
            continue
        n, deb, k, mn, mx = max(ok)
        print(f"{conf} : palier retenu = {n} tâche(s) simultanée(s), {deb:.1f} jetons/s par GPU "
              f"(médiane de {k} répétition(s), de {mn:.1f} à {mx:.1f}).")
        if n == paliers[-1]:
            print(f"   ATTENTION : c'est le plus haut palier testé ; ajoute des paliers au-dessus de {n}.")
        if any(r < n for r in refus):
            print(f"   ATTENTION : des paliers plus bas ({', '.join(str(r) for r in refus if r < n)}) ne respectent pas la cible : vérifie ces essais.")
        if sans:
            print("   ATTENTION : métriques du serveur absentes pour certains paliers : préemptions non vérifiées.")
        lr = groupes[(conf, n)]
        if any((x["requetes_en_attente_max"] or 0) > 0 for x in lr):
            print("   ATTENTION : des requêtes ont attendu dans la file du serveur à ce palier.")
        if any((x["cache_kv_max_fraction"] or 0) >= 0.95 for x in lr):
            print("   ATTENTION : cache KV rempli à 95 % ou plus à ce palier : la mémoire est presque saturée.")


if __name__ == "__main__":
    main()
