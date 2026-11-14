"""Relevé des métriques Prometheus de vLLM pendant un palier des agents de test.

Interroge une ou plusieurs URL /metrics toutes les `pas_s` secondes entre `t_debut` et `t_fin`,
et calcule, sur cette fenêtre : la variation des compteurs (préemptions, cache de préfixes,
décodage spéculatif) et le maximum de l'occupation du cache KV. Plusieurs URL (plusieurs répliques)
sont additionnées ; pour les jauges (cache KV), on garde le maximum par série (un moteur par série quand
DP > 1). Aucune clé n'est envoyée : vLLM ne protège pas /metrics. Bibliothèque standard seulement.
"""
import re, threading, time, urllib.request

COMPTEURS = {
    "preemptions": "vllm:num_preemptions",
    "cache_prefixes_reussites": "vllm:prefix_cache_hits",
    "cache_prefixes_requetes": "vllm:prefix_cache_queries",
    "mtp_acceptes": "vllm:spec_decode_num_accepted_tokens",
    "mtp_proposes": "vllm:spec_decode_num_draft_tokens",
}
JAUGE_KV = "vllm:kv_cache_usage_perc"
JAUGE_ATTENTE = "vllm:num_requests_waiting"
_LIGNE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfa]+)")


def lire(url, delai_s=5, maxima=None):
    """Valeurs d'un /metrics, additionnées par nom (le suffixe _total des compteurs est retiré).
    Si `maxima` est un dict, il reçoit le maximum par série de chaque nom."""
    with urllib.request.urlopen(urllib.request.Request(url), timeout=delai_s) as r:
        if r.geturl() != url:
            raise ValueError("redirection refusée")
        texte = r.read().decode("utf-8", "replace")
    vals = {}
    for ligne in texte.splitlines():
        m = _LIGNE.match(ligne)
        if not m:
            continue
        nom = m.group(1)
        if nom.endswith("_total"):
            nom = nom[:-6]
        try:
            x = float(m.group(3))
        except ValueError:
            continue
        vals[nom] = vals.get(nom, 0.0) + x
        if maxima is not None:
            maxima[nom] = x if nom not in maxima else max(maxima[nom], x)
    return vals


class Releve:
    """Relevé en arrière-plan ; resultat() donne les variations sur la fenêtre et le cache KV maximal."""

    def __init__(self, urls, t_debut, t_fin, pas_s=2.0):
        self.urls, self.t_debut, self.t_fin, self.pas = urls, t_debut, t_fin, pas_s
        self.premier, self.dernier, self.kv_max, self.erreurs, self.lectures = {}, {}, None, 0, 0
        self.attente_max = None
        self._fil = threading.Thread(target=self._boucle, daemon=True)

    def demarrer(self):
        if self.urls:
            self._fil.start()
        return self

    def _instantane(self):
        total, kv = {}, None
        for u in self.urls:
            mx = {}
            v = lire(u, maxima=mx)
            for k, x in v.items():
                total[k] = total.get(k, 0.0) + x
            if JAUGE_KV in mx:
                kv = mx[JAUGE_KV] if kv is None else max(kv, mx[JAUGE_KV])
        return total, kv

    def _boucle(self):
        while time.time() < self.t_debut:
            time.sleep(min(self.pas, max(0.0, self.t_debut - time.time())))
        while time.time() <= self.t_fin:
            try:
                total, kv = self._instantane()
                if not self.premier:
                    self.premier = total
                self.dernier = total
                if kv is not None:
                    self.kv_max = kv if self.kv_max is None else max(self.kv_max, kv)
                if JAUGE_ATTENTE in total:
                    a = total[JAUGE_ATTENTE]
                    self.attente_max = a if self.attente_max is None else max(self.attente_max, a)
                self.lectures += 1
            except Exception:  # noqa: BLE001 — une lecture manquée n'arrête pas le palier
                self.erreurs += 1
            time.sleep(self.pas)

    def resultat(self):
        if self.urls and self._fil.is_alive():
            self._fil.join(timeout=self.pas * 3)
        r = {"urls": self.urls, "lectures": self.lectures, "lectures_echouees": self.erreurs,
             "cache_kv_max_fraction": self.kv_max, "requetes_en_attente_max": self.attente_max}
        for cle, nom in COMPTEURS.items():
            r[cle] = (self.dernier[nom] - self.premier[nom]) if self.lectures >= 2 and nom in self.dernier and nom in self.premier else None
        return r
