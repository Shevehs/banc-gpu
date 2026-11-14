"""Lecture de /metrics (format Prometheus) et indicateurs d'une fenêtre d'observation.
Bibliothèque standard seulement. Utilisé par observer.py et depouiller_observation.py."""
import math, re, urllib.error, urllib.parse, urllib.request

_LIGNE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

# Noms au commit vLLM 7f855f6 (vllm/v1/metrics/loggers.py) ; le suffixe _total des compteurs est retiré.
COMPTEURS = {
    "jetons_sortie": "vllm:generation_tokens",
    "jetons_entree": "vllm:prompt_tokens",
    "requetes_terminees": "vllm:request_success",
    "preemptions": "vllm:num_preemptions",
    "cache_prefixes_reussites": "vllm:prefix_cache_hits",
    "cache_prefixes_requetes": "vllm:prefix_cache_queries",
    "mtp_acceptes": "vllm:spec_decode_num_accepted_tokens",
    "mtp_proposes": "vllm:spec_decode_num_draft_tokens",
    "requetes_en_erreur": "http_requests",
}
HISTOS = {
    "ttft": "vllm:time_to_first_token_seconds",
    # moyenne par requête du temps entre jetons, premier jeton exclu (vllm/v1/metrics/stats.py)
    "tpot": "vllm:request_time_per_output_token_seconds",
}
JAUGES = {"kv": "vllm:kv_cache_usage_perc", "attente": "vllm:num_requests_waiting",
          "en_cours": "vllm:num_requests_running"}


class _RedirectionBarre(urllib.request.HTTPRedirectHandler):
    """Ne suit qu'une redirection : même hôte, même chemin plus « / » (LiteLLM sert /metrics/, et
    redirige /metrics vers /metrics/). Toute autre redirection est refusée, avant tout envoi de la clé."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        a, b = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(urllib.parse.urljoin(req.full_url, newurl))
        if (a.scheme, a.netloc) == (b.scheme, b.netloc) and b.path == a.path.rstrip("/") + "/":
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        raise urllib.error.HTTPError(req.full_url, code, f"redirection refusée vers {newurl}", headers, fp)


# Pas de proxy : les /metrics sont sur le réseau privé, et la clé de LiteLLM ne doit pas passer par un proxy.
_OUVREUR = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RedirectionBarre)
ERREURS_HTTP = "http_requests"   # compteur de prometheus_fastapi_instrumentator exposé par vLLM


def lire(url, delai_s=5, prefixe=("vllm:", ERREURS_HTTP), cle=None):
    """Liste de (nom, étiquettes, valeur) d'un /metrics, pour les noms qui commencent par `prefixe`.
    `cle` n'est donnée que pour le /metrics de LiteLLM (protégé par défaut)."""
    req = urllib.request.Request(url)
    if cle:
        req.add_header("Authorization", f"Bearer {cle}")
    with _OUVREUR.open(req, timeout=delai_s) as r:
        texte = r.read().decode("utf-8", "replace")
    res = []
    for ligne in texte.splitlines():
        m = _LIGNE.match(ligne)
        if not m or not m.group(1).startswith(prefixe):
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        nom = m.group(1)
        if nom.endswith("_total"):
            nom = nom[:-6]
        res.append((nom, dict(_LABEL.findall(m.group(2) or "")), v))
    return res


def resumer(series):
    """Compteurs sommés par nom, histogrammes sommés par borne, jauges par série."""
    compteurs, histos, jauges = {}, {k: {} for k in HISTOS}, {k: [] for k in JAUGES}
    noms_c = set(COMPTEURS.values()) - {ERREURS_HTTP}
    for nom, lab, v in series:
        if nom in noms_c:
            compteurs[nom] = compteurs.get(nom, 0.0) + v
        # Requêtes d'inférence en erreur (4xx, 5xx) : http_requests_total{handler="/v1/…", status="4xx"|"5xx"}
        # (prometheus_fastapi_instrumentator, codes groupés par défaut ; INFÉRÉ, à vérifier sur la version installée).
        # Une série d'erreurs n'apparaît qu'à la première erreur : dès qu'une série /v1 existe, le compte part de 0.
        if nom == ERREURS_HTTP and lab.get("handler", "").startswith("/v1/"):
            compteurs.setdefault(ERREURS_HTTP, 0.0)
            if lab.get("status", "")[:1] in ("4", "5"):
                compteurs[ERREURS_HTTP] += v
        for k, h in HISTOS.items():
            if nom == h + "_bucket" and "le" in lab:
                histos[k][lab["le"]] = histos[k].get(lab["le"], 0.0) + v
        for k, j in JAUGES.items():
            if nom == j:
                jauges[k].append(v)
    return compteurs, histos, jauges


def _borne(le):
    return math.inf if le in ("+Inf", "Inf", "inf") else float(le)


def cumul(h):
    """[(borne, cumul)] trié, à partir d'un dict {le: compte}."""
    return sorted(((_borne(k), v) for k, v in h.items()), key=lambda x: x[0])


def quantile(h, q):
    """Borne haute du seau qui contient le quantile q (valeur prudente : le vrai quantile est au plus là).
    Renvoie (borne, au-delà) ; au-delà = le quantile dépasse la dernière borne finie (renvoyée)."""
    c = cumul(h)
    if not c or c[-1][1] <= 0:
        return None, False
    total = c[-1][1]; cible = q * total; bas = 0.0
    for b, n in c:
        if n >= cible:
            return (bas, True) if math.isinf(b) else (b, False)
        bas = b
    return bas, True


def part_sous(h, x):
    """Part des observations ≤ x : (estimation, exacte, minimum, maximum).
    Exacte si une borne tombe à moins de 1 % de x ; sinon interpolée, entre minimum (cumul à la borne
    inférieure) et maximum (cumul à la borne supérieure). (None, None, None, None) si aucune observation."""
    c = cumul(h)
    if not c or c[-1][1] <= 0:
        return None, None, None, None
    total = c[-1][1]
    for b, n in c:
        if not math.isinf(b) and abs(b - x) <= 0.01 * x:
            return n / total, True, n / total, n / total
    bas, c_bas = 0.0, 0.0
    for b, n in c:
        if b >= x:
            if math.isinf(b):
                return c_bas / total, False, c_bas / total, 1.0
            return (c_bas + (n - c_bas) * (x - bas) / (b - bas)) / total, False, c_bas / total, n / total
        bas, c_bas = b, n
    return 1.0, False, 1.0, 1.0


def indicateurs(obs, cible, gpu, ttft_max_ms=None):
    """Indicateurs d'une fenêtre (dict lu dans observation.json)."""
    d = obs["fin"] - obs["debut"]; c = obs["compteurs"]; h = obs["histogrammes"]
    rap = lambda a, b: a / b if a is not None and b else None
    sortie = c.get("jetons_sortie")
    tpot90, au_dela = quantile(h["tpot"], 0.90)  # borne haute -> interactivité P10 au moins 1/borne
    part, exacte, part_min, part_max = part_sous(h["tpot"], 1.0 / cible)
    ttft50, _ = quantile(h["ttft"], 0.50); ttft95, ttft_au_dela = quantile(h["ttft"], 0.95)
    r = {
        "requetes_terminees": c.get("requetes_terminees"),
        "debit_sortie_par_gpu": sortie / d / gpu if sortie is not None and d > 0 else None,
        "part_requetes_a_la_cible": part, "part_exacte": exacte, "part_min": part_min, "part_max": part_max,
        "requetes_en_erreur": c.get("requetes_en_erreur"),
        "interactivite_p10_estimee": (1.0 / tpot90) if tpot90 else None, "interactivite_borne": au_dela,
        "ttft_p50_ms_estime": ttft50 * 1000 if ttft50 is not None else None,
        "ttft_p95_ms_estime": ttft95 * 1000 if ttft95 is not None else None, "ttft_p95_au_dela": ttft_au_dela,
        "preemptions": c.get("preemptions"),
        "cache_prefixes": rap(c.get("cache_prefixes_reussites"), c.get("cache_prefixes_requetes")),
        "acceptation_mtp": rap(c.get("mtp_acceptes"), c.get("mtp_proposes")),
        "cache_kv_max_fraction": obs["jauges"].get("kv_max"),
        "requetes_en_attente_max": obs["jauges"].get("attente_max"),
    }
    # Verdict à trois valeurs : True (tient), False (ne tient pas), None (indéterminé : la cible tombe dans un
    # seau qui encadre 90 %, ou aucune requête). Erreurs et préemptions font échouer ; TTFT seulement si demandé.
    if (r["preemptions"] or 0) > 0 or (r["requetes_en_erreur"] or 0) > 0:
        r["tient"] = False
    elif part is None:
        r["tient"] = None
    elif part_max < 0.90:
        r["tient"] = False
    elif ttft_max_ms is not None and (r["ttft_p95_ms_estime"] is None or ttft_au_dela
                                      or r["ttft_p95_ms_estime"] > ttft_max_ms):
        r["tient"] = False
    else:
        r["tient"] = True if part_min >= 0.90 else None
    return r


def verdict(t):
    return {True: "tient", False: "ne tient pas", None: "indéterminé"}[t]


# ---- Vue par agent : métriques Prometheus de LiteLLM (callback « prometheus »), ventilées par une étiquette
# (api_key_alias = une clé virtuelle par agent, ou metadata_<champ> = custom_prometheus_metadata_labels).
# Noms d'après https://docs.litellm.ai/docs/proxy/prometheus ; vérifiés avec LiteLLM 1.109.1 lors de la validation.
LITELLM = {
    "jetons_sortie": "litellm_output_tokens_metric",
    "jetons_entree": "litellm_input_tokens_metric",
    "ttft": "litellm_llm_api_time_to_first_token_metric",     # histogramme (s), requêtes en streaming seulement
    "latence": "litellm_request_total_latency_metric",        # histogramme (s), de bout en bout
}


def resumer_litellm(series, etiquette):
    """{agent: {jetons_sortie, jetons_entree, ttft{le}, ttft_sum, ttft_count, latence{le}, latence_sum, latence_count}}"""
    res = {}
    for nom, lab, v in series:
        ag = lab.get(etiquette)
        if ag is None:
            continue
        a = res.setdefault(ag, {"jetons_sortie": 0.0, "jetons_entree": 0.0, "ttft": {}, "ttft_sum": 0.0,
                                "ttft_count": 0.0, "latence": {}, "latence_sum": 0.0, "latence_count": 0.0})
        for k in ("jetons_sortie", "jetons_entree"):
            if nom == LITELLM[k]:
                a[k] += v
        for k in ("ttft", "latence"):
            if nom == LITELLM[k] + "_bucket" and "le" in lab:
                a[k][lab["le"]] = a[k].get(lab["le"], 0.0) + v
            elif nom == LITELLM[k] + "_sum":
                a[k + "_sum"] += v
            elif nom == LITELLM[k] + "_count":
                a[k + "_count"] += v
    return res


def delta_litellm(debut, fin):
    """Variation par agent entre deux résumés ; un agent absent au début part de zéro."""
    res = {}
    for ag, b in fin.items():
        a = debut.get(ag) or {}
        d = {}
        for k, v in b.items():
            if isinstance(v, dict):
                d[k] = {le: n - (a.get(k) or {}).get(le, 0.0) for le, n in v.items()}
            else:
                d[k] = v - (a.get(k) or 0.0)
        if d["latence_count"] > 0 or d["jetons_sortie"] > 0:
            res[ag] = d
    return res


def indicateurs_agents(agents):
    """Indicateurs par agent. Vitesse moyenne = Σ sortie ÷ (Σ latence − Σ TTFT), seulement si toutes les
    requêtes de l'agent sont en streaming (sinon le TTFT manque) : moyenne, pas un centile (INFÉRÉ)."""
    total_sortie = sum(a["jetons_sortie"] for a in agents.values()) or None
    res = {}
    for ag, a in sorted(agents.items()):
        n = a["latence_count"]
        t50, _ = quantile(a["ttft"], 0.50); t95, t95_au_dela = quantile(a["ttft"], 0.95)
        l50, _ = quantile(a["latence"], 0.50); l95, l95_au_dela = quantile(a["latence"], 0.95)
        decode = a["latence_sum"] - a["ttft_sum"]
        res[ag] = {
            "requetes": n,
            "part_jetons_sortie": a["jetons_sortie"] / total_sortie if total_sortie else None,
            "jetons_entree_par_requete": a["jetons_entree"] / n if n else None,
            "jetons_sortie_par_requete": a["jetons_sortie"] / n if n else None,
            "ttft_p50_s": t50, "ttft_p95_s": t95, "ttft_p95_au_dela": t95_au_dela,
            "latence_p50_s": l50, "latence_p95_s": l95, "latence_p95_au_dela": l95_au_dela,
            "vitesse_moyenne": (a["jetons_sortie"] / decode) if n and a["ttft_count"] == n and decode > 0 else None,
        }
    return res
