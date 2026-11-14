"""Dépouille les essais AIPerf : un tableau par palier et le point de fonctionnement retenu.
Usage : python depouiller.py resultats --cible 30 [--gpu 8] [--exiger-metriques]
  --cible : interactivité visée par fil (jetons/s) ; le 10e centile doit l'atteindre
            (interactivité d'une requête = 1 / délai moyen entre jetons, hors premier jeton)
  --gpu   : nombre de GPU derrière la cible, toutes répliques comprises (débit par GPU = débit total / gpu) ;
            facultatif : sans lui, déduit de config_serveur.json de chaque essai (TP × PP × DP × répliques)
  --exiger-metriques : refuse un palier dont les métriques du serveur (préemptions) n'ont pas été lues
  --ttft-max : refuse un palier dont le TTFT P95 dépasse cette valeur (ms) ; à fixer selon l'usage (HYPOTHÈSE)
Colonne err : requêtes en erreur, plus les requêtes refusées pour contexte trop long (AIPerf les retire de
ses métriques ; un palier qui en compte est refusé).
Écrit <racine>/tableau.csv et affiche le point retenu par configuration (essais <conf>-r1, -r2…).
Bibliothèque standard seulement."""
import argparse, csv, json, pathlib, re, statistics


def stat(d, cle, champ):
    v = d.get(cle) or {}
    return v.get(champ) if isinstance(v, dict) else None


def debord(d, cle="context_overflow_count"):
    v = d.get(cle)
    v = v.get("avg") if isinstance(v, dict) else v
    return v or 0


def nombre_gpu(dossier_essai, gpu_donne, notes):
    """GPU derrière la cible pour un essai : --gpu s'il est donné, sinon déduit de son config_serveur.json
    (TP × PP × DP de l'empreinte × répliques). Note l'origine dans `notes` ; None si rien ne permet de le savoir."""
    import sys as _s
    _s.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "outils"))
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


def serveur(dossier):
    """Compteurs du serveur vLLM sur la phase de mesure (deltas calculés par AIPerf), et cache KV maximal."""
    f = dossier / "server_metrics_export.json"
    if not f.exists():
        return {}
    m = json.loads(f.read_text()).get("metrics", {})

    def total(nom):
        v = m.get(nom)
        if not v:
            return None
        return sum((s.get("stats") or {}).get("total") or 0 for s in v.get("series", []))

    def maxi(nom):
        v = m.get(nom)
        if not v:
            return None
        xs = [(s.get("stats") or {}).get("max") for s in v.get("series", [])]
        xs = [x for x in xs if x is not None]
        return max(xs) if xs else None

    r = {"preemptions": total("vllm:num_preemptions"), "cache_kv_max": maxi("vllm:kv_cache_usage_perc"),
         "attente_max": maxi("vllm:num_requests_waiting")}
    h, q = total("vllm:prefix_cache_hits"), total("vllm:prefix_cache_queries")
    r["cache_prefixes"] = h / q if h is not None and q else None
    a, dr = total("vllm:spec_decode_num_accepted_tokens"), total("vllm:spec_decode_num_draft_tokens")
    r["acceptation_mtp"] = a / dr if a is not None and dr else None
    return r


def main():
    a = argparse.ArgumentParser()
    a.add_argument("racine"); a.add_argument("--cible", type=float, required=True)
    a.add_argument("--gpu", type=int, default=None, help="GPU derrière la cible ; défaut : déduit de l'empreinte")
    a.add_argument("--exiger-metriques", action="store_true")
    a.add_argument("--ttft-max", type=float, default=None, help="TTFT P95 maximal accepté (ms)")
    p = a.parse_args()
    if p.gpu is not None and p.gpu < 1:
        raise SystemExit("--gpu : entier ≥ 1")
    lignes, notes_gpu, sans_gpu = [], {}, {}
    for f in sorted(pathlib.Path(p.racine).glob("*/c*/profile_export_aiperf.json")):
        d = json.loads(f.read_text()); dossier = f.parent
        etiq = dossier.parent.name; conc = int(dossier.name[1:])
        n_gpu = nombre_gpu(dossier.parent, p.gpu, notes_gpu)
        if n_gpu is None:
            sans_gpu[etiq] = notes_gpu.pop(etiq, "?"); continue
        s = serveur(dossier)
        debit = stat(d, "output_token_throughput", "avg")
        n_err = stat(d, "error_request_count", "avg")
        if n_err is None:
            n_err = 0 if not d.get("error_summary") else sum(e.get("count", 1) for e in d["error_summary"])
        lignes.append({
            "configuration": re.sub(r"-r\d+$", "", etiq), "essai": etiq, "concurrence": conc,
            "requetes": stat(d, "request_count", "avg"), "requetes_en_erreur": int(n_err),
            "sortie_moyenne_jetons": stat(d, "output_sequence_length", "avg"),
            # Requêtes refusées pour contexte trop long : AIPerf les retire des métriques (scénario AgentX)
            "debordements_contexte": int(debord(d)),
            # Sorties plus courtes que demandé (vLLM réduit max_tokens à la place restante du contexte)
            "sorties_raccourcies": int(debord(d, "osl_mismatch_count")),
            "essai_valide_aiperf": (d.get("metadata") or {}).get("submission_valid"),
            "raisons_invalide": ";".join((d.get("metadata") or {}).get("submission_invalid_reasons") or []),
            "debit_sortie_total": debit,
            "gpu": n_gpu, "debit_sortie_par_gpu": debit / n_gpu if debit else None,
            "interactivite_p10": stat(d, "output_token_throughput_per_user", "p10"),
            "interactivite_p50": stat(d, "output_token_throughput_per_user", "p50"),
            "ttft_p50_ms": stat(d, "time_to_first_token", "p50"),
            "ttft_p95_ms": stat(d, "time_to_first_token", "p95"),
            "ttft_p99_ms": stat(d, "time_to_first_token", "p99"),
            "preemptions": s.get("preemptions"), "cache_kv_max_fraction": s.get("cache_kv_max"),
            "cache_prefixes": s.get("cache_prefixes"), "acceptation_mtp": s.get("acceptation_mtp"),
            "requetes_en_attente_max": s.get("attente_max")})
    lignes.sort(key=lambda l: (l["essai"], l["concurrence"]))
    # Essais qu'AIPerf ne tient pas pour valides (scénario AgentX) et filtre de contexte noté par agentx.sh
    for etiq in sorted({l["essai"] for l in lignes}):
        ls = [l for l in lignes if l["essai"] == etiq]
        if any(l["essai_valide_aiperf"] is False for l in ls):
            raisons = sorted({r for l in ls for r in l["raisons_invalide"].split(";") if r})
            print(f"ATTENTION : {etiq} : AIPerf marque l'essai submission_valid=false ({', '.join(raisons) or '?'}) : "
                  "non comparable aux résultats AgentX publiés.")
        racc = [str(l["concurrence"]) for l in ls if l["sorties_raccourcies"]]
        if racc:
            print(f"ATTENTION : {etiq} : sorties plus courtes que demandé aux concurrences {', '.join(racc)} "
                  "(osl_mismatch_count d'AIPerf : contexte presque plein, vLLM réduit max_tokens) : débit et "
                  "interactivité portent sur des sorties tronquées ; monte --max-model-len ou baisse CONTEXTE_MAX.")
        meta = pathlib.Path(p.racine) / etiq / "agentx.txt"
        if meta.exists():
            info = dict(x.split("=", 1) for x in meta.read_text().splitlines() if "=" in x)
            if info.get("contexte_max"):
                print(f"ATTENTION : {etiq} : traces filtrées à {info['contexte_max']} jetons (CONTEXTE_MAX) : sous-ensemble.")
    if sans_gpu:
        raise SystemExit("ERREUR : nombre de GPU inconnu pour "
                         + " ; ".join(f"{e} ({r})" for e, r in sorted(sans_gpu.items()))
                         + " : donne --gpu (champ « GPU » dans l'interface).")
    for n in notes_gpu.values():
        print(n)
    par_conf = {}
    for l in lignes:
        par_conf.setdefault(re.sub(r"-r\d+$", "", l["essai"]), {})[l["essai"]] = l["gpu"]
    for conf, ess in sorted(par_conf.items()):
        if len(set(ess.values())) > 1:
            print(f"ATTENTION : {conf} : nombre de GPU différent selon les répétitions ("
                  + ", ".join(f"{e} = {g}" for e, g in sorted(ess.items()))
                  + ") : la médiane mélange des configurations ; vérifie, ou donne --gpu.")
    if not lignes:
        raise SystemExit(f"Aucun profile_export_aiperf.json sous {p.racine}/<essai>/c<N>/")
    sortie = pathlib.Path(p.racine) / "tableau.csv"
    with open(sortie, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0])); w.writeheader(); w.writerows(lignes)
    print(f"Tableau : {sortie}\n")
    f2 = lambda x, n=1: "—" if x is None else f"{x:.{n}f}"
    pc = lambda x: "—" if x is None else f"{100 * x:.0f}%"
    print(f"{'essai':<18}{'conc':>5}{'débit/GPU':>11}{'inter P10':>11}{'TTFT P95 ms':>13}{'préempt':>9}{'err':>5}{'cache':>7}{'MTP':>6}{'sortie moy':>12}")
    for l in lignes:
        print(f"{l['essai']:<18}{l['concurrence']:>5}{f2(l['debit_sortie_par_gpu']):>11}{f2(l['interactivite_p10']):>11}"
              f"{f2(l['ttft_p95_ms'], 0):>13}{f2(l['preemptions'], 0):>9}{l['requetes_en_erreur'] + l['debordements_contexte']:>5}"
              f"{pc(l['cache_prefixes']):>7}{pc(l['acceptation_mtp']):>6}{f2(l['sortie_moyenne_jetons'], 0):>12}")
    # Point de fonctionnement : médiane des répétitions par (configuration, concurrence), puis le palier
    # au débit le plus élevé qui respecte la cible, sans requête en erreur ni préemption.
    groupes = {}
    for l in lignes:
        groupes.setdefault((l["configuration"], l["concurrence"]), []).append(l)
    print()
    for conf in sorted({k[0] for k in groupes}):
        paliers = sorted(c for (cf, c) in groupes if cf == conf)
        ok, refus, sans_metriques, autre_motif = [], [], False, set()
        for c in paliers:
            ls = groupes[(conf, c)]
            p10 = statistics.median([l["interactivite_p10"] or 0 for l in ls])
            debs = [l["debit_sortie_par_gpu"] or 0 for l in ls]
            err = sum(l["requetes_en_erreur"] + l["debordements_contexte"] for l in ls)  # débordements = erreurs
            manque = any(l["preemptions"] is None for l in ls)
            sans_metriques |= manque
            pre = sum(l["preemptions"] or 0 for l in ls)
            ttft = statistics.median([l["ttft_p95_ms"] or 0 for l in ls])
            sain = (err == 0 and pre == 0 and not (manque and p.exiger_metriques)
                    and (p.ttft_max is None or ttft <= p.ttft_max))
            if not sain:
                autre_motif.add(c)  # refusé pour une autre raison que l'interactivité
            if p10 >= p.cible and sain:
                ok.append((statistics.median(debs), c, len(ls), min(debs), max(debs)))
            else:
                refus.append(c)
        if not ok:
            print(f"{conf} : aucun palier ne respecte la cible de {p.cible} jetons/s sans erreur ni préemption.")
            continue
        deb, conc, n, mn, mx = max(ok)
        print(f"{conf} : point retenu = concurrence {conc}, {deb:.1f} jetons/s par GPU "
              f"(médiane de {n} répétition(s), de {mn:.1f} à {mx:.1f}).")
        if conc == paliers[-1]:
            print(f"   ATTENTION : c'est le plus haut palier testé. La cible n'est peut-être pas encore atteinte : "
                  f"ajoute des paliers au-dessus de {conc}.")
        # Interpolation linéaire entre le palier retenu et le palier suivant (qui ne respecte pas la cible),
        # dans le plan interactivité P10 -> débit par GPU : estimation du débit exactement à la cible (INFÉRÉ).
        suivants = [c for c in paliers if c > conc]
        if suivants and suivants[0] not in autre_motif:  # seulement si le suivant n'échoue que sur la cible
            a_, b_ = groupes[(conf, conc)], groupes[(conf, suivants[0])]
            x1 = statistics.median([l["interactivite_p10"] or 0 for l in a_]); y1 = deb
            x2 = statistics.median([l["interactivite_p10"] or 0 for l in b_])
            y2 = statistics.median([l["debit_sortie_par_gpu"] or 0 for l in b_])
            if x1 > p.cible > x2 and x1 != x2 and y2 > y1:
                yi = y1 + (y2 - y1) * (x1 - p.cible) / (x1 - x2)
                print(f"   Estimation à exactement {p.cible} jetons/s (interpolation entre les concurrences {conc} et "
                      f"{suivants[0]}) : {yi:.1f} jetons/s par GPU. À confirmer par un palier intermédiaire.")
        if any(c < conc for c in refus):
            print(f"   ATTENTION : des paliers plus bas ({', '.join(str(c) for c in refus if c < conc)}) ne respectent pas "
                  f"la cible : courbe non monotone, vérifie ces essais.")
        if sans_metriques:
            print("   ATTENTION : métriques du serveur absentes pour certains paliers : préemptions non vérifiées.")
        lr = groupes[(conf, conc)]
        if any((l["requetes_en_attente_max"] or 0) > 0 for l in lr):
            print("   ATTENTION : des requêtes ont attendu dans la file du serveur à ce palier : le TTFT inclut de l'attente.")
        if any((l["cache_kv_max_fraction"] or 0) >= 0.95 for l in lr):
            print("   ATTENTION : cache KV rempli à 95 % ou plus à ce palier : la mémoire est presque saturée.")


if __name__ == "__main__":
    main()
