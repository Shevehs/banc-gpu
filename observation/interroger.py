"""Observation de l'usage réel : interroge Prometheus, qui collecte /metrics de vLLM (et de LiteLLM), et
donne un verdict par fenêtre, par période ou par agent. Ne génère aucune charge et n'a rien à lancer :
Prometheus garde l'historique ; on interroge n'importe quelle plage passée.

Usage : python observation/interroger.py --prometheus http://127.0.0.1:9090 --cible 30 --gpu 8
        [--vllm 'job="vllm"'] [--litellm 'job="litellm"' --par metadata_agent]
        [--depuis 2h | --debut "2026-11-14 09:00" [--fin "2026-11-14 17:00"]] [--fenetre 5m]
        [--vue fenetres|periode|agents] [--periode plage|heure] [--ttft-max ms]
        [--exclure "2026-11-14 10:00/2026-11-14 11:00" ...] [--collecte 15s] [--sortie tableau.csv]
  --vllm, --litellm : sélecteurs d'étiquettes Prometheus des cibles (job de prometheus.yml). Toutes les
            séries qui correspondent sont additionnées : un sélecteur = une configuration (toutes répliques).
  --cible : interactivité visée par requête (jetons/s). Une fenêtre « tient » si au moins 90 % des requêtes
            terminées l'atteignent (même seuil que l'interactivité P10 des stress tests, mesuré ici par vLLM),
            sans préemption ni requête en erreur (et avec un TTFT P95 ≤ --ttft-max si donné). « Indéterminé » :
            la cible tombe dans un seau d'histogramme qui encadre 90 % (bornes à ajouter, README § 6.1).
  --gpu   : nombre de GPU derrière le sélecteur vLLM, toutes répliques comprises
  --exclure : plages à écarter (stress test sur le même serveur : la charge synthétique se mêlerait à l'usage réel)
  --collecte : intervalle de collecte de Prometheus (scrape_interval), pour les jauges
Clé : variable BANC_CLE_PROMETHEUS si Prometheus exige un jeton (Bearer). Bibliothèque standard seulement.
"""
import argparse, csv, json, math, os, re, sys, time, urllib.error, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse import COMPTEURS, HISTOS, LITELLM, cumul, indicateurs, indicateurs_agents, verdict  # noqa: E402

PLAGES = [(0, "nuit (0-6 h)"), (6, "matin (6-12 h)"), (12, "après-midi (12-18 h)"), (18, "soir (18-24 h)")]
SELECTEUR = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*(=|!=|=~|!~)\s*"[^"\\]*"'
                       r'(\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*(=|!=|=~|!~)\s*"[^"\\]*")*\s*$')
ETIQUETTE = re.compile(r"^(api_key_alias|team_alias|user|end_user|metadata_[A-Za-z0-9_]{1,40})$")
f = lambda x, n=1: "—" if x is None or x == "" else f"{float(x):.{n}f}"
pc = lambda x: "—" if x is None or x == "" else f"{100 * float(x):.0f}%"


class _SansRedirection(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # une redirection est une erreur : la clé ne part pas ailleurs
        return None


# Pas de proxy : Prometheus est sur le réseau privé, et sa clé ne doit pas passer par un proxy.
_OUVREUR = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SansRedirection)


def duree(v):
    """'300', '300s', '5m', '2h', '1d' -> secondes."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", str(v))
    if not m:
        raise SystemExit(f"durée invalide : {v} (ex. 300, 5m, 2h, 1d)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def instant(v):
    """'AAAA-MM-JJ HH:MM' (heure locale) ou horodatage Unix -> secondes."""
    v = str(v).strip()
    if re.match(r"^\d+(\.\d+)?$", v):
        return float(v)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return time.mktime(time.strptime(v, fmt))
        except ValueError:
            pass
    raise SystemExit(f"date invalide : {v} (forme AAAA-MM-JJ HH:MM, heure locale)")


def selecteur(v, quoi):
    v = (v or "").strip()
    if not SELECTEUR.match(v):
        raise SystemExit(f"{quoi} : sélecteur d'étiquettes attendu, ex. job=\"vllm\" ou job=\"vllm\",instance=~\"10.0.0.*\"")
    return v


class Prometheus:
    def __init__(self, base, cle=None, delai_s=30):
        self.base, self.cle, self.delai_s = base.rstrip("/"), cle, delai_s

    def plage(self, q, debut, fin, pas):
        """[(étiquettes, {horodatage: valeur})] pour une requête PromQL évaluée de debut à fin, tous les pas."""
        url = self.base + "/api/v1/query_range?" + urllib.parse.urlencode(
            {"query": q, "start": f"{debut:.3f}", "end": f"{fin:.3f}", "step": f"{pas:g}"})
        req = urllib.request.Request(url)
        if self.cle:
            req.add_header("Authorization", f"Bearer {self.cle}")
        try:
            with _OUVREUR.open(req, timeout=self.delai_s) as r:
                j = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            corps = e.read().decode("utf-8", "replace")[:300] if e.fp else ""
            raise SystemExit(f"Prometheus a répondu HTTP {e.code} à « {q[:120]} » : {corps}")
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(f"Prometheus injoignable ({self.base}) : {e}")
        if j.get("status") != "success":
            raise SystemExit(f"Prometheus : {j.get('error') or j}")
        return [(s.get("metric") or {}, {round(float(t)): float(v) for t, v in s.get("values") or []
                                          if v not in ("NaN", "+Inf", "-Inf")})
                for s in (j.get("data") or {}).get("result") or []]


def accroissement(serie, w, par=""):
    """Accroissement d'un compteur sur chaque fenêtre ]t-w, t], additionné (par étiquettes `par`).
    - Valeur à t moins valeur à t-w (dernières collectes avant ces instants) : exact, et des fenêtres successives
      s'additionnent sans perte ni double compte.
    - Série apparue pendant la fenêtre (premier code d'erreur 5xx, premier appel d'un agent dans LiteLLM) : sa
      valeur entière, si sa cible était déjà collectée au début de la fenêtre (sinon, par exemple au premier
      lancement de Prometheus, la valeur cumulée depuis le démarrage du serveur fausserait la fenêtre).
    Une remise à zéro (redémarrage) rend la différence fausse : ces fenêtres sont repérées par redemarrages()
    et écartées."""
    W = f"{int(w)}s"
    exact = f"(({serie} - {serie} offset {W}) >= 0)"
    nouvelle = f"(({serie} unless {serie} offset {W}) and on(job, instance) (up offset {W} == 1))"
    q = f"{exact} or {nouvelle}"
    return f"sum by ({par}) ({q})" if par else f"sum({q})"


TEMOINS_VLLM = ["vllm:request_success_total", "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
                "vllm:request_time_per_output_token_seconds_count", "http_requests_total"]
TEMOINS_LITELLM = ["litellm_request_total_latency_metric_count", "litellm_output_tokens_metric_total",
                   "litellm_input_tokens_metric_total"]


def redemarrages(sel, noms, w, collecte):
    """Nombre d'indices de redémarrage dans la fenêtre (plus une collecte avant, pour voir le passage de la
    frontière) : baisse d'un compteur témoin (resets) ou changement de process_start_time_seconds (le seul indice quand
    le compteur a dépassé sa valeur d'avant entre deux collectes ; exposé par prometheus_client sous Linux)."""
    R = f"{int(w + collecte)}s"
    termes = [f"(sum(resets({n}{sel}[{R}])) or vector(0))" for n in noms]
    termes.append(f"(sum(changes(process_start_time_seconds{sel}[{R}])) or vector(0))")
    return " + ".join(termes)


def releve(prom, sel_v, debut, fin, w, collecte, sel_l=None, par=None):
    """Une observation par fenêtre : {debut, fin, compteurs, histogrammes, jauges, cibles, agents}."""
    fins = []
    t = debut + w
    while t <= fin + 1e-6:
        fins.append(round(t)); t += w
        if len(fins) > 10_000:
            raise SystemExit("Plus de 10 000 fenêtres (Prometheus en refuse au-delà de 11 000 points par requête) : "
                             "allonge --fenetre ou raccourcis la plage.")
    if not fins:
        raise SystemExit("Plage plus courte qu'une fenêtre : allonge --depuis ou raccourcis --fenetre.")
    a, b = fins[0], fins[-1]
    W, C = f"{int(w)}s", f"{int(collecte)}s"
    un = lambda q: (prom.plage(q, a, b, w) or [({}, {})])[0][1]
    par_le = lambda q: {s.get("le"): v for s, v in prom.plage(q, a, b, w)}
    sel = "{" + sel_v + "}"
    remises = un(redemarrages(sel, TEMOINS_VLLM, w, collecte))
    collectes = un(f"sum(count_over_time(up{sel}[{W}]))")
    err_sel = "{" + sel_v + ',handler=~"/v1/.*",status=~"[45].*"}'
    compteurs = {k: un(accroissement(n + (err_sel if k == "requetes_en_erreur" else sel), w))
                 for k, n in COMPTEURS.items()}
    expose_err = un(f"count(http_requests_total{sel})")
    histos = {k: par_le(accroissement(n + "_bucket" + sel, w, "le")) for k, n in HISTOS.items()}
    jauges = {
        "kv_max": un(f"max_over_time(max(vllm:kv_cache_usage_perc{sel})[{W}:{C}])"),
        "attente_max": un(f"max_over_time(sum(vllm:num_requests_waiting{sel})[{W}:{C}])"),
        "en_cours_moy": un(f"avg_over_time(sum(vllm:num_requests_running{sel})[{W}:{C}])"),
        "en_cours_max": un(f"max_over_time(sum(vllm:num_requests_running{sel})[{W}:{C}])"),
    }
    cibles_min = un(f"min_over_time(sum(up{sel})[{W}:{C}])")
    cibles = un(f"max_over_time(count(up{sel})[{W}:{C}])")
    cibles_debut = un(f"sum(up{sel} offset {W})")  # cibles joignables à la dernière collecte avant le début
    # Cibles collectées dans la collecte qui précède le début (]t-w-1,5C, t-w]) : sans elle, « X offset W »
    # remonterait jusqu'à 5 min en arrière (lookback de Prometheus) et la fenêtre compterait la coupure.
    R0 = f"{int(1500 * collecte)}ms"
    cibles_frontiere = un(f"count(count_over_time(up{sel}[{R0}] offset {W}))")
    # Idem à la fin (]t-1,5C, t]) : sans collecte récente, la valeur de fin serait ancienne (fenêtre tronquée).
    cibles_fin = un(f"count(count_over_time(up{sel}[{R0}]))")
    agents_t, cibles_l = None, None
    if sel_l:
        sl = "{" + sel_l + "}"
        cibles_l = (un(f"min_over_time(sum(up{sl})[{W}:{C}])"), un(f"max_over_time(count(up{sl})[{W}:{C}])"),
                    un(redemarrages(sl, TEMOINS_LITELLM, w, collecte)))
        agents_t = {}
        for k in ("jetons_sortie", "jetons_entree"):
            for lab, vals in prom.plage(accroissement(LITELLM[k] + sl, w, par), a, b, w):
                for t, v in vals.items():
                    agents_t.setdefault(t, {}).setdefault(lab.get(par, ""), {})[k] = v
        for k in ("ttft", "latence"):
            for lab, vals in prom.plage(accroissement(LITELLM[k] + "_bucket" + sl, w, par + ", le"), a, b, w):
                for t, v in vals.items():
                    agents_t.setdefault(t, {}).setdefault(lab.get(par, ""), {}).setdefault(k, {})[lab.get("le")] = v
            for suf in ("sum", "count"):
                for lab, vals in prom.plage(accroissement(LITELLM[k] + "_" + suf + sl, w, par), a, b, w):
                    for t, v in vals.items():
                        agents_t.setdefault(t, {}).setdefault(lab.get(par, ""), {})[f"{k}_{suf}"] = v
    obs = []
    for t in fins:
        c = {k: v.get(t) for k, v in compteurs.items()}
        if c["requetes_en_erreur"] is None and expose_err.get(t):
            c["requetes_en_erreur"] = 0.0  # séries http_requests_total présentes, aucune /v1 en 4xx/5xx
        o = {"debut": t - w, "fin": t, "compteurs": c,
             "histogrammes": {k: {le: v[t] for le, v in h.items() if t in v} for k, h in histos.items()},
             "jauges": {k: v.get(t) for k, v in jauges.items()},
             "cibles": cibles.get(t), "cibles_toujours_up": cibles_min.get(t), "cibles_debut": cibles_debut.get(t),
             "remises": remises.get(t) or 0, "collectes": collectes.get(t),
             "cibles_frontiere": cibles_frontiere.get(t) or 0, "cibles_fin": cibles_fin.get(t) or 0}
        if sel_l:
            o["litellm_cibles"] = (cibles_l[1].get(t), cibles_l[0].get(t))
            o["litellm_remises"] = cibles_l[2].get(t) or 0
            ags = {}
            for ag, d in (agents_t.get(t) or {}).items():
                d = {"jetons_sortie": 0.0, "jetons_entree": 0.0, "ttft": {}, "ttft_sum": 0.0, "ttft_count": 0.0,
                     "latence": {}, "latence_sum": 0.0, "latence_count": 0.0, **d}
                if d["latence_count"] > 0 or d["jetons_sortie"] > 0:
                    ags[ag] = d
            o["agents"] = ags
        obs.append(o)
    return obs


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
    """Additionne des fenêtres successives (compteurs, histogrammes ; jauges : maximum, concurrence : moyenne)."""
    c = {}
    for o in obs:
        for k, v in o["compteurs"].items():
            if k == "requetes_en_erreur":  # None = aucune série /v1 encore créée : rien à ajouter
                c[k] = c.get(k) if v is None else (c.get(k) or 0.0) + v
            else:  # autre compteur illisible dans une fenêtre : somme inconnue
                c[k] = None if v is None or (k in c and c[k] is None) else c.get(k, 0.0) + v
    j = {k: max((o["jauges"][k] for o in obs if o["jauges"][k] is not None), default=None) for k in obs[0]["jauges"]}
    moy = [o["jauges"]["en_cours_moy"] for o in obs if o["jauges"]["en_cours_moy"] is not None]
    j["en_cours_moy"] = sum(moy) / len(moy) if moy else None
    return {"debut": 0.0, "fin": sum(o["fin"] - o["debut"] for o in obs), "compteurs": c, "jauges": j,
            "histogrammes": {k: somme_histo([o["histogrammes"][k] for o in obs]) for k in obs[0]["histogrammes"]}}


def part_txt(r):
    if r["part_requetes_a_la_cible"] is None:
        return "—"
    return pc(r["part_requetes_a_la_cible"]) + ("" if r["part_exacte"] else "*")


def entete():
    return (f"{'conc moy':>9}{'requêtes':>9}{'err':>5}{'débit/GPU':>11}{'à la cible':>11}{'inter P10≥':>11}"
            f"{'TTFT P95≤ ms':>13}{'préempt':>8}{'cache':>7}{'MTP':>6}{'KV max':>8}  verdict")


def ligne(r):
    return (f"{f(r['concurrence_moyenne']):>9}{f(r['requetes_terminees'], 0):>9}{f(r['requetes_en_erreur'], 0):>5}"
            f"{f(r['debit_sortie_par_gpu']):>11}{part_txt(r):>11}"
            f"{('<' if r['interactivite_borne'] else '') + f(r['interactivite_p10_estimee']):>11}"
            f"{('>' if r['ttft_p95_au_dela'] else '') + f(r['ttft_p95_ms_estime'], 0):>13}{f(r['preemptions'], 0):>8}"
            f"{pc(r['cache_prefixes']):>7}{pc(r['acceptation_mtp']):>6}{pc(r['cache_kv_max_fraction']):>8}"
            f"  {r['verdict']}")


def croissant(h):
    """Vrai si les comptes cumulés d'un histogramme croissent avec la borne (sinon : données incohérentes)."""
    c = cumul(h)
    return all(b[1] >= a[1] - 1e-9 for a, b in zip(c, c[1:]))


def evaluer(o, p):
    r = indicateurs(o, p.cible, p.gpu, p.ttft_max)
    if not all(croissant(h) for h in o["histogrammes"].values()):
        for k in ("part_requetes_a_la_cible", "part_min", "part_max", "interactivite_p10_estimee",
                  "ttft_p50_ms_estime", "ttft_p95_ms_estime"):
            r[k] = None
        r["tient"], r["histogramme_incoherent"] = None, True
    r["concurrence_moyenne"] = o["jauges"].get("en_cours_moy")
    r["concurrence_max"] = o["jauges"].get("en_cours_max")
    r["verdict"] = "aucune requête" if r["requetes_terminees"] == 0 else verdict(r["tient"])
    return r


def main():
    a = argparse.ArgumentParser(description="Observation de l'usage réel à partir de Prometheus.")
    a.add_argument("--prometheus", required=True, help="URL de Prometheus, ex. http://127.0.0.1:9090")
    a.add_argument("--vllm", default='job="vllm"', help='sélecteur des cibles vLLM (défaut job="vllm")')
    a.add_argument("--litellm", default=None, help='sélecteur des cibles LiteLLM, ex. job="litellm" (vue par agent)')
    a.add_argument("--par", default="metadata_agent", help="étiquette LiteLLM qui distingue les agents")
    a.add_argument("--depuis", default="2h", help="plage se terminant maintenant (ex. 30m, 2h, 1d) ; ignoré avec --debut")
    a.add_argument("--debut", default=None); a.add_argument("--fin", default=None)
    a.add_argument("--fenetre", default="5m", help="durée d'une fenêtre (ex. 5m, 15m, 1h)")
    a.add_argument("--collecte", default="15s", help="scrape_interval de Prometheus")
    a.add_argument("--cible", type=float, required=True); a.add_argument("--gpu", type=int, required=True)
    a.add_argument("--ttft-max", type=float, default=None, help="TTFT P95 estimé maximal accepté (ms)")
    a.add_argument("--vue", choices=["fenetres", "periode", "agents"], default="fenetres")
    a.add_argument("--periode", choices=["heure", "plage"], default="plage")
    a.add_argument("--exclure", action="append", default=[], help="plage « début/fin » à écarter (stress test)")
    a.add_argument("--sortie", default=None, help="CSV des fenêtres (optionnel)")
    p = a.parse_args()
    sel_v = selecteur(p.vllm, "--vllm")
    sel_l = selecteur(p.litellm, "--litellm") if p.litellm else None
    if sel_l and not ETIQUETTE.match(p.par):
        raise SystemExit("--par : api_key_alias, team_alias, user, end_user ou metadata_<champ>")
    if p.vue == "agents" and not sel_l:
        raise SystemExit("--vue agents : donne --litellm (sélecteur des cibles LiteLLM dans Prometheus).")
    w, collecte = duree(p.fenetre), duree(p.collecte)
    if w < 4 * collecte:
        raise SystemExit(f"--fenetre doit valoir au moins 4 × --collecte ({4 * collecte:g} s).")
    fin = instant(p.fin) if p.fin else time.time()
    debut = instant(p.debut) if p.debut else fin - duree(p.depuis)
    if p.debut and not p.fin:
        fin = time.time()
    fin = min(fin, time.time())
    exclues = []
    for x in p.exclure:
        if "/" not in x:
            raise SystemExit(f"--exclure : forme « début/fin » attendue ({x})")
        d0, d1 = x.split("/", 1); exclues.append((instant(d0), instant(d1)))
    prom = Prometheus(p.prometheus, os.environ.get("BANC_CLE_PROMETHEUS") or None)
    obs = releve(prom, sel_v, debut, fin, w, collecte, sel_l, p.par if sel_l else None)

    lignes, alertes, valides = [], [], []
    for o in obs:
        quand = time.strftime("%Y-%m-%d %H:%M", time.localtime(o["debut"]))
        r = evaluer(o, p)
        motif = None
        if not o["cibles"]:
            motif = "aucune cible vLLM collectée (sélecteur, ou Prometheus pas encore lancé)"
        elif (o["cibles_debut"] or 0) < o["cibles"]:
            motif = "cible pas encore collectée ou injoignable au début de la fenêtre"
        elif o["cibles_frontiere"] < o["cibles"] or o["cibles_fin"] < o["cibles"]:
            motif = ("pas de collecte juste avant le début ou la fin de la fenêtre (Prometheus arrêté, ou --collecte "
                     "plus court que son scrape_interval) : différence faussée")
        elif (o["cibles_toujours_up"] or 0) < o["cibles"]:
            motif = "une cible vLLM injoignable pendant la fenêtre (collecte manquée)"
        elif o["remises"] > 0:
            motif = "vLLM redémarré pendant la fenêtre (compteurs remis à zéro)"
        elif (o["collectes"] or 0) < 0.8 * o["cibles"] * w / collecte:
            motif = (f"collectes manquantes ({o['collectes'] or 0:.0f} pour ≈ {o['cibles'] * w / collecte:.0f} attendues ; "
                     "Prometheus arrêté, ou --collecte plus court que son scrape_interval)")
        elif any(d0 < o["fin"] and o["debut"] < d1 for d0, d1 in exclues):
            motif = "plage exclue (stress test)"
        if motif:
            r = {k: (None if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in r.items()}
            r["verdict"] = "essai en cours" if motif.startswith("plage") else "écartée"
            if not motif.startswith("aucune cible vLLM"):
                alertes.append(f"{quand} : {motif} : fenêtre écartée.")
        else:
            valides.append(o)
            if o["collectes"] and o["cibles"] and o["collectes"] > 1.2 * o["cibles"] * w / collecte:
                reel = o["cibles"] * w / o["collectes"]
                alertes.append(f"--collecte ({collecte:g} s) plus long que l'intervalle réel de Prometheus "
                               f"(≈ {reel:.0f} s) : contrôles de collecte et jauges faussés ; donne --collecte {reel:.0f}s.")
            if r.get("histogramme_incoherent"):
                alertes.append(f"{quand} : histogramme incohérent (seaux non croissants) : verdict indéterminé.")
            if (r["requetes_terminees"] or 0) and r["requetes_terminees"] < 50:
                alertes.append(f"{quand} : moins de 50 requêtes terminées : centiles peu fiables.")
        if sel_l and o.get("litellm_cibles") and (o["litellm_cibles"][1] or 0) < (o["litellm_cibles"][0] or 0):
            alertes.append(f"{quand} : une cible LiteLLM injoignable pendant la fenêtre : écartée de la vue par agent.")
            o["agents"] = None
        if sel_l and o.get("litellm_remises"):
            alertes.append(f"{quand} : LiteLLM redémarré pendant la fenêtre : écartée de la vue par agent.")
            o["agents"] = None
        if sel_l and o.get("agents") and not all(croissant(d[k]) for d in o["agents"].values() for k in ("ttft", "latence")):
            alertes.append(f"{quand} : histogramme LiteLLM incohérent (redémarrage non repéré ?) : écartée de la vue par agent.")
            o["agents"] = None
        if sel_l and o["cibles"] and not (o.get("litellm_cibles") or (None,))[0]:
            alertes.append(f"{quand} : aucune cible LiteLLM collectée (sélecteur {sel_l}).")
        lignes.append({"debut": quand, "fin": time.strftime("%H:%M", time.localtime(o["fin"])),
                       "duree_s": round(o["fin"] - o["debut"]), "cibles": o["cibles"], **r, "ecartee": bool(motif)})
    if all(not o["cibles"] for o in obs):
        raise SystemExit(f"Aucune cible vLLM {{{sel_v}}} dans Prometheus sur cette plage : vérifie le sélecteur "
                         "(job de prometheus.yml) et la page Status > Targets de Prometheus.")
    if p.sortie:
        with open(p.sortie, "w", newline="", encoding="utf-8") as fh:
            champs = list(dict.fromkeys(k for l in lignes for k in l))
            wr = csv.DictWriter(fh, fieldnames=champs); wr.writeheader(); wr.writerows(lignes)
    print(f"Prometheus {p.prometheus}, vLLM {{{sel_v}}}"
          + (f", LiteLLM {{{sel_l}}} par {p.par}" if sel_l else "")
          + f" ; {time.strftime('%Y-%m-%d %H:%M', time.localtime(obs[0]['debut']))} → "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(obs[-1]['fin']))}, fenêtres de {w:g} s\n")

    if p.vue == "periode":
        groupes = {}
        for o in valides:
            groupes.setdefault(periode(o["debut"], p.periode), []).append(o)
        print(f"{'période':<22}{'fenêtres':>9}" + entete())
        for per in sorted(groupes):
            print(f"{per:<22}{len(groupes[per]):>9}" + ligne(evaluer(fusionner(groupes[per]), p)))
        print("\nDébit/GPU : moyenne sur la période ; conc moy : requêtes en cours en moyenne. Pointe d'une période : "
              "--vue fenetres.")
    elif p.vue == "agents":
        tous = {}
        for o in valides:
            for ag, d in (o.get("agents") or {}).items():
                t = tous.setdefault(ag, {k: ({} if isinstance(v, dict) else 0.0) for k, v in d.items()})
                for k, v in d.items():
                    t[k] = somme_histo([t[k], v]) if isinstance(v, dict) else t[k] + v
        if not tous:
            print(f"Aucune requête par agent sur la plage (LiteLLM {{{sel_l}}}, étiquette {p.par}).")
        else:
            print(f"  {'agent':<24}{'requêtes':>9}{'part sortie':>12}{'entrée/req':>11}{'sortie/req':>11}"
                  f"{'TTFT P50≤ s':>12}{'TTFT P95≤ s':>12}{'latence P95≤ s':>15}{'vitesse moy':>12}")
            for ag, r in indicateurs_agents(tous).items():
                print(f"  {(ag if ag not in ('', 'None') else '(sans identité)')[:24]:<24}{f(r['requetes'], 0):>9}{pc(r['part_jetons_sortie']):>12}"
                      f"{f(r['jetons_entree_par_requete'], 0):>11}{f(r['jetons_sortie_par_requete'], 0):>11}"
                      f"{f(r['ttft_p50_s'], 2):>12}{('>' if r['ttft_p95_au_dela'] else '') + f(r['ttft_p95_s'], 2):>12}"
                      f"{('>' if r['latence_p95_au_dela'] else '') + f(r['latence_p95_s'], 2):>15}{f(r['vitesse_moyenne']):>12}")
            if "" in tous or "None" in tous:
                print(f"   (sans identité) : requêtes sans valeur pour l'étiquette {p.par}.")
        print("\nVue par agent lue côté LiteLLM : TTFT seulement pour les requêtes en streaming ; vitesse moyenne = jetons "
              "de sortie ÷ (latence − TTFT), moyenne et non centile, temps du proxy compris, vide si une partie des "
              "requêtes n'est pas en streaming. Bornes ≤ : seaux des histogrammes de LiteLLM.")
    else:
        print(f"{'début':<17}{'fin':>6}" + entete())
        for l in lignes:
            print(f"{l['debut']:<17}{l['fin']:>6}" + ligne(l))
        if any(l["part_exacte"] is False for l in lignes):
            print(f"\n* part interpolée : aucune borne d'histogramme à {1000 / p.cible:.1f} ms ; lance vLLM avec "
                  "--custom-histogram-buckets (README, § 6.1). « indéterminé » : la cible tombe dans un seau qui encadre 90 %.")
        print("≥ et ≤ : bornes lues dans les seaux des histogrammes (< ou > : au-delà de la dernière borne). "
              "err : requêtes /v1 en 4xx ou 5xx vues par vLLM (« — » si non exposé). conc moy : requêtes en cours, en moyenne.")
        tenues = [l for l in lignes if l["verdict"] == "tient" and (l["requetes_terminees"] or 0) >= 50
                  and (l["concurrence_moyenne"] or 0) > 0]
        ratees = [l for l in lignes if l["verdict"] == "ne tient pas" and l["concurrence_moyenne"] is not None]
        print()
        if tenues:
            m = max(tenues, key=lambda l: l["concurrence_moyenne"])
            print(f"Plus haute concurrence moyenne tenue : {m['concurrence_moyenne']:.1f} requêtes en cours "
                  f"({m['debut']}, {f(m['debit_sortie_par_gpu'])} jetons/s par GPU).")
            plus = [l for l in ratees if l["concurrence_moyenne"] <= m["concurrence_moyenne"]]
            if plus:
                print("   ATTENTION : des fenêtres à concurrence égale ou plus basse ne tiennent pas "
                      f"({', '.join(l['debut'] for l in plus[:5])}) : voir err, préemptions et TTFT.")
            indet = [l for l in lignes if l["verdict"] == "indéterminé"
                     and (l["concurrence_moyenne"] or 0) > m["concurrence_moyenne"]]
            if indet:
                print(f"   ATTENTION : {len(indet)} fenêtre(s) plus chargée(s) au verdict indéterminé "
                      "(bornes d'histogramme, README § 6.1) : la limite peut être plus haute.")
            if not [l for l in ratees if l["concurrence_moyenne"] > m["concurrence_moyenne"]]:
                print("   ATTENTION : aucune fenêtre plus chargée qui échoue : la limite n'est pas atteinte sur cette plage.")
        else:
            print(f"Aucune fenêtre d'au moins 50 requêtes ne tient {p.cible:g} jetons/s pour 90 % des requêtes "
                  "sans préemption ni erreur (ou résultat indéterminé : voir les verdicts).")
        if any((l["requetes_en_attente_max"] or 0) > 0 for l in tenues):
            print("   ATTENTION : des requêtes ont attendu dans la file du serveur dans des fenêtres qui tiennent.")
        if any((l["cache_kv_max_fraction"] or 0) >= 0.95 for l in lignes if not l["ecartee"]):
            print("   ATTENTION : cache KV rempli à 95 % ou plus dans au moins une fenêtre.")
    for al in dict.fromkeys(alertes):
        print("   ATTENTION : " + al)


if __name__ == "__main__":
    main()
