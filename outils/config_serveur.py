"""Instantané de la configuration du serveur vLLM, à enregistrer avec chaque essai.

Lit, sans rien modifier :
  - sur la cible : /v1/models (modèle, longueur de contexte maximale si la cible est vLLM) et, si le modèle
    est donné, une complétion d'un seul jeton pour lire system_fingerprint. En mode par défaut de vLLM
    (--fingerprint-mode full), cette empreinte vaut vllm-<version>[-tpN][-ppN][-dpN][-ep]-<hash8> :
    elle donne le parallélisme sans mode développement ; le hash change avec la configuration.
    En mode --fingerprint-mode hash, les degrés sont masqués : l'empreinte ressemble alors à celle d'un
    serveur sans parallélisme (« 1 ou masqué »).
  - sur chaque serveur vLLM (URL de métriques sans /metrics) : /version ; dans /metrics, vllm:cache_config_info
    (cache de préfixes, type et taille du cache) et le nombre de positions de
    vllm:spec_decode_num_accepted_tokens_per_pos (= jetons spéculatifs par étape) ; /server_info, qui n'existe
    que si vLLM a été lancé avec VLLM_SERVER_DEV_MODE=1 (méthode de décodage spéculatif, ordonnanceur…).
Un budget de temps (45 s par défaut) plafonne le délai de chaque requête et arrête les suivantes une fois
épuisé ; un serveur qui répond au goutte-à-goutte peut toutefois le dépasser. La complétion d'un jeton vient
en dernier, pour qu'une cible lente n'empêche pas le reste du relevé.
La clé d'API n'est envoyée qu'à la cible : /metrics, /version et /server_info ne la demandent pas
(vLLM ne protège que /v1, /v2, /inference et /cohere). Les redirections ne sont pas suivies.

Usage : python outils/config_serveur.py --url http://hôte:8000 [--modele banc]
        [--metriques "http://h1:8000/metrics …"] [--sortie config_serveur.json]
Clé d'API : variable CLE_API ou BANC_CLE_API. Bibliothèque standard seulement.
"""
import argparse, json, os, re, sys, time, urllib.error, urllib.request

_INFO = re.compile(r'^vllm:cache_config_info\{([^}]*)\}')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_POSITION = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?\{[^}]*position="(\d+)"')
_EMPREINTE = re.compile(r'^vllm-(?P<version>.+?)(?P<par>(?:-(?:tp|pp|dp)\d+|-ep)*)-(?P<hash>[0-9a-zA-Z]{8}|nohash)$')


class _SansRedirection(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # une redirection est traitée comme une erreur
        return None


# Pas de proxy : la cible et les /metrics sont sur le réseau privé, et la clé ne doit pas passer par un proxy.
_OUVREUR = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SansRedirection)


def _requete(url, cle=None, corps=None, delai_s=5):
    req = urllib.request.Request(url, data=json.dumps(corps).encode() if corps is not None else None)
    if corps is not None:
        req.add_header("Content-Type", "application/json")
    if cle:
        req.add_header("Authorization", f"Bearer {cle}")
    with _OUVREUR.open(req, timeout=delai_s) as r:
        return r.status, r.read(2_000_000).decode("utf-8", "replace")


def _json(url, cle=None, corps=None, delai_s=5):
    try:
        s, t = _requete(url, cle, corps, delai_s)
        j = json.loads(t) if s == 200 else None
        return j if isinstance(j, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _d(x):
    return x if isinstance(x, dict) else {}


def lire_empreinte(e):
    """Décode system_fingerprint (mode full de vLLM). Degré absent = 1 (ou masqué en mode hash)."""
    if not isinstance(e, str) or len(e) > 300:
        return None
    m = _EMPREINTE.match(e)
    if not m:
        return None
    par = m.group("par")
    deg = lambda k: int(re.search(rf"-{k}(\d+)", par).group(1)) if re.search(rf"-{k}(\d+)", par) else 1
    return {"version": m.group("version"), "tensor_parallel_size": deg("tp"), "pipeline_parallel_size": deg("pp"),
            "data_parallel_size": deg("dp"), "enable_expert_parallel": "-ep" in par, "hash_config": m.group("hash"),
            "masque_possible": not par}  # aucun segment : TP=PP=DP=1 sans EP, ou mode hash qui masque tout


def _configuration(cfg):
    """Extrait les champs utiles de vllm_config (format JSON de /server_info)."""
    if not isinstance(cfg, dict):
        return None
    par, mod, cache, sched = (_d(cfg.get(k)) for k in ("parallel_config", "model_config", "cache_config",
                                                         "scheduler_config"))
    spec = _d(cfg.get("speculative_config"))
    garde = lambda d, ks: {k: d.get(k) for k in ks if k in d}
    return {
        "parallelisme": garde(par, ["tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                                    "enable_expert_parallel", "decode_context_parallel_size",
                                    "prefill_context_parallel_size"]),
        "modele": garde(mod, ["model", "served_model_name", "dtype", "quantization", "max_model_len"]),
        "cache": garde(cache, ["cache_dtype", "enable_prefix_caching", "gpu_memory_utilization", "block_size",
                               "num_gpu_blocks"]),
        "ordonnanceur": garde(sched, ["max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill",
                                      "async_scheduling"]),
        "decodage_speculatif": garde(spec, ["method", "num_speculative_tokens"]) if spec else None,
    }


def releve(url, metriques=None, cle=None, modele=None, budget_s=45):
    """Dictionnaire décrivant la cible et chaque serveur vLLM. Ne lève pas d'exception sur une réponse inattendue."""
    fin = time.time() + budget_s
    reste = lambda d: max(1.0, min(d, fin - time.time()))
    url = url.rstrip("/")
    urls_m = list(metriques or []) or [url + "/metrics"]
    res = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cible": url, "metriques_donnees": bool(metriques),
           "serveurs": [], "empreinte": None}
    modeles = _json(url + "/v1/models", cle, delai_s=reste(5))
    donnees = modeles.get("data") if modeles else None
    res["modeles_cible"] = [{"id": m.get("id"), "max_model_len": m.get("max_model_len")}
                            for m in donnees if isinstance(m, dict)] if isinstance(donnees, list) else None
    for um in urls_m:
        base = um[: -len("/metrics")] if um.endswith("/metrics") else um
        srv = {"metriques": um, "version": None, "cache_config": None, "jetons_speculatifs_metriques": None,
               "configuration": None, "lignes_vllm": None, "statut_metriques": None}
        if time.time() < fin:
            v = _json(base + "/version", delai_s=reste(5))
            srv["version"] = v.get("version") if v else None
        try:
            if time.time() >= fin:
                srv["statut_metriques"] = "budget"
            s, t = _requete(um, delai_s=reste(5)) if time.time() < fin else (0, "")
            if s:
                srv["statut_metriques"] = s
            positions, lignes_vllm = set(), 0
            for ligne in t.splitlines() if s == 200 else []:
                lignes_vllm += ligne.startswith("vllm:")
                m = _INFO.match(ligne)
                if m and srv["cache_config"] is None:
                    srv["cache_config"] = {k: v for k, v in _LABEL.findall(m.group(1))
                                           if k in ("cache_dtype", "enable_prefix_caching", "gpu_memory_utilization",
                                                    "block_size", "num_gpu_blocks", "engine")}
                p = _POSITION.match(ligne)
                if p:
                    positions.add(p.group(1))
            srv["lignes_vllm"] = lignes_vllm if s == 200 else None
            if s == 200 and lignes_vllm:
                srv["jetons_speculatifs_metriques"] = len(positions)  # 0 = pas de décodage spéculatif (INFÉRÉ)
        except urllib.error.HTTPError as e:
            srv["statut_metriques"] = e.code  # 401/403 : adresse protégée par une clé (LiteLLM), pas celle de vLLM
        except Exception:  # noqa: BLE001
            pass
        # Premier appel parfois lent (vLLM collecte l'environnement système) : délai plus long.
        info = _json(base + "/server_info?config_format=json", delai_s=reste(20)) if time.time() < fin else None
        srv["configuration"] = _configuration(info.get("vllm_config")) if info else None
        res["serveurs"].append(srv)
    if modele and time.time() >= fin:
        res["empreinte"] = {"brute": None, "decodee": None, "reponse": None, "budget_epuise": True}
    elif modele:  # une complétion d'un jeton, en dernier et avant toute mesure
        r = _json(url + "/v1/chat/completions", cle, {"model": modele, "max_tokens": 1, "temperature": 0,
                                                      "messages": [{"role": "user", "content": "ok"}]},
                  delai_s=reste(15))
        e = r.get("system_fingerprint") if r else None
        res["empreinte"] = {"brute": e if isinstance(e, str) else None,
                            "decodee": lire_empreinte(e),
                            "reponse": r is not None}
    return res


def gpu_deduits(r):
    """Nombre de GPU derrière la cible, déduit du relevé : TP × PP × DP lus dans l'empreinte, multiplié par le
    nombre de serveurs vLLM relevés (une URL de métriques par réplique ; toutes supposées identiques à celle qui
    a répondu). Renvoie (nombre ou None, explication, masque_possible)."""
    try:
        dec = (r.get("empreinte") or {}).get("decodee")
        if not dec:
            return None, "empreinte absente (complétion d'un jeton non faite, ou --fingerprint-mode none ou custom)", False
        # Serveur vLLM reconnu : des lignes vllm: dans /metrics (relevés plus anciens : version ou métriques lues)
        serveurs = [s for s in (r.get("serveurs") or [])
                    if ((s.get("lignes_vllm") or 0) > 0 if "lignes_vllm" in s
                        else s.get("jetons_speculatifs_metriques") is not None or bool(s.get("cache_config")))]
        if not serveurs:
            return None, "aucune URL de métriques ne répond comme un serveur vLLM (METRIQUES vide ou fausse)", False
        tp, pp, dp = (int(dec[k]) for k in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"))
        par = ((serveurs[0].get("configuration") or {}).get("parallelisme") or {})
        pcp = int(par.get("prefill_context_parallel_size") or 1)  # absent de l'empreinte ; lu si /server_info existe
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return None, f"config_serveur.json de forme inattendue ({type(e).__name__})", False
    n = tp * pp * dp * pcp * len(serveurs)
    return n, (f"{n} (TP {tp} × PP {pp} × DP {dp}" + (f" × PCP {pcp}" if pcp > 1 else "")
               + f" × {len(serveurs)} serveur(s) vLLM)"), bool(dec.get("masque_possible"))


def gpu_essai(dossier):
    """gpu_deduits() pour le config_serveur.json d'un dossier d'essai ; (None, raison, False) s'il manque."""
    import pathlib
    f = pathlib.Path(dossier) / "config_serveur.json"
    if not f.exists():
        return None, "config_serveur.json absent", False
    try:
        return gpu_deduits(json.loads(f.read_text(encoding="utf-8")))
    except (ValueError, OSError) as e:
        return None, f"config_serveur.json illisible ({e})", False


def resume(r):
    """Une ligne par serveur, pour les journaux."""
    dec = (r.get("empreinte") or {}).get("decodee")
    lignes = []
    if (r.get("empreinte") or {}).get("budget_epuise"):
        lignes.append("[config] empreinte non relevée : budget de temps épuisé (serveurs lents)")
    elif r.get("empreinte"):
        lignes.append(f"[config] empreinte de la cible : {r['empreinte'].get('brute') or 'absente'}"
                      + (f" -> TP {dec['tensor_parallel_size']}, PP {dec['pipeline_parallel_size']}, "
                         f"DP {dec['data_parallel_size']}, EP {'oui' if dec['enable_expert_parallel'] else 'non'}"
                         + (" (ou masqué si --fingerprint-mode hash)" if dec["masque_possible"] else "")
                         if dec else "")
                      + ("" if r["empreinte"].get("reponse") else " (pas de réponse : modèle, clé ou cible à vérifier)"))
    n, expl, masque = gpu_deduits(r)
    if n:
        lignes.append(f"[config] GPU déduits : {expl}" + (" ; 1 GPU par serveur sauf si --fingerprint-mode hash masque "
                                                          "le parallélisme" if masque else ""))
    for s in r["serveurs"]:
        cc = s["cache_config"] or {}
        sp = (s["configuration"] or {}).get("decodage_speculatif")
        js = s.get("jetons_speculatifs_metriques")
        lignes.append(f"[config] {s['metriques']} : vLLM {s['version'] or '?'} ; cache de préfixes "
                      f"{cc.get('enable_prefix_caching', '?')} ; jetons spéculatifs (métriques) "
                      f"{js if js is not None else '?'} ; /server_info "
                      + (f"lu (décodage spéculatif : {sp or 'aucun'})" if s["configuration"] else
                         "absent ou non lu (VLLM_SERVER_DEV_MODE non activé ?)"))
        st = s.get("statut_metriques")
        if s.get("lignes_vllm") == 0 or st in (401, 403, 307, 308):
            lignes.append(f"[config] ATTENTION : {s['metriques']} n'est pas le /metrics de vLLM ("
                          + ("aucune métrique vllm:" if s.get("lignes_vllm") == 0 else f"HTTP {st} : clé demandée")
                          + ", adresse de LiteLLM ?) : préemptions, cache et cache KV ne seront pas lus ; mets l'adresse "
                          "/metrics de vLLM dans METRIQUES (ex. http://<hôte-vllm>:8000/metrics).")
        elif st == "budget":
            lignes.append(f"[config] ATTENTION : {s['metriques']} non lu (budget de temps épuisé, serveurs lents).")
        elif s.get("lignes_vllm") is None:
            lignes.append(f"[config] ATTENTION : {s['metriques']} injoignable"
                          + (f" (HTTP {st})" if isinstance(st, int) else "")
                          + " : préemptions, cache et cache KV ne seront pas lus.")
    return "\n".join(lignes)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--url", required=True); a.add_argument("--metriques", default="")
    a.add_argument("--modele", default=None, help="si donné, une complétion d'un jeton pour lire l'empreinte")
    a.add_argument("--sortie", default=None)
    p = a.parse_args()
    cle = os.environ.get("BANC_CLE_API") or os.environ.get("CLE_API") or None
    r = releve(p.url, p.metriques.split(), cle, p.modele)
    texte = json.dumps(r, indent=1, ensure_ascii=False)
    if p.sortie:
        with open(p.sortie, "w", encoding="utf-8") as f:
            f.write(texte)
    else:
        print(texte)
    print(resume(r), file=sys.stderr)


if __name__ == "__main__":
    main()
