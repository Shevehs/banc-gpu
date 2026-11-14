"""Outils communs aux agents de test : appel au modèle mesuré, journal par appel, contenu synthétique.

Aucun outil n'agit sur un vrai système : toutes les « sorties d'outils » sont fabriquées ici,
de façon déterministe à partir d'une graine. Rien ne sort de la machine, sauf les appels au
serveur du modèle (URL donnée au lanceur).
"""
import json, os, pathlib, random, threading, time

from openai import OpenAI

# ---------------------------------------------------------------- journal
_verrou = threading.Lock()


class Journal:
    """Écrit une ligne JSON par appel au modèle (appels.jsonl) et par tâche (taches.jsonl)."""

    def __init__(self, dossier):
        self.f_appels = open(f"{dossier}/appels.jsonl", "a", encoding="utf-8")
        self.f_taches = open(f"{dossier}/taches.jsonl", "a", encoding="utf-8")

    def appel(self, **ligne):
        with _verrou:
            self.f_appels.write(json.dumps(ligne) + "\n"); self.f_appels.flush()

    def tache(self, **ligne):
        with _verrou:
            self.f_taches.write(json.dumps(ligne) + "\n"); self.f_taches.flush()

    def fermer(self):
        self.f_appels.close(); self.f_taches.close()


# ---------------------------------------------------------------- contenu synthétique
# Taille visée en jetons. Si un tokenizer est disponible (fichier tokenizer.json dans BANC_TOKENIZER,
# ou outils/tokenizer/ du dépôt), la taille est ajustée sur le vrai compte de jetons. Sinon, on
# estime avec un nombre de caractères par jeton propre à chaque nature de texte, étalonné sur le
# vocabulaire BPE de Qwen (INFÉRÉ : même vocabulaire que Qwen3.8) : les chiffres comptent un
# jeton chacun, d'où des rapports bas pour les journaux et les scans.
CAR_PAR_JETON = {"code": 3.0, "scan": 2.0, "journal": 1.9, "notes": 3.2}

_MOTS = ("buffer index length offset parse header token session config request response "
         "handler payload decode encode user input output value check bound size alloc free "
         "path file socket port host service version module function return error status").split()
_TOK = {"charge": False, "tok": None}


def tokenizer():
    """Tokenizer local (bibliothèque tokenizers), ou None."""
    if not _TOK["charge"]:
        _TOK["charge"] = True
        racine = os.environ.get("BANC_TOKENIZER") or os.environ.get("TOKENIZER") or str(pathlib.Path(__file__).resolve().parent.parent / "outils" / "tokenizer")
        f = pathlib.Path(racine) / "tokenizer.json" if pathlib.Path(racine).is_dir() else pathlib.Path(racine)
        if f.is_file():
            try:
                from tokenizers import Tokenizer
                _TOK["tok"] = Tokenizer.from_file(str(f))
            except Exception:  # noqa: BLE001 — sans tokenizer, on garde l'estimation
                _TOK["tok"] = None
    return _TOK["tok"]


def _fabriquer(nature, n_car, graine):
    r = random.Random(graine)
    lignes, n, i = [], 0, 0
    while n < n_car:
        i += 1
        m = [r.choice(_MOTS) for _ in range(6)]
        if nature == "code":
            l = f"int {m[0]}_{i}(char *{m[1]}, size_t {m[2]}) {{ if ({m[2]} > {r.randint(8, 4096)}) return -1; memcpy({m[3]}, {m[1]}, {m[2]}); }} // {m[4]} {m[5]}"
        elif nature == "scan":
            l = f"10.0.{r.randint(0, 255)}.{r.randint(1, 254)}  {r.choice([22, 80, 443, 445, 3389, 8080])}/tcp open  {m[0]}-{m[1]} v{r.randint(1, 9)}.{r.randint(0, 20)}  ({m[2]} {m[3]})"
        elif nature == "journal":
            l = f"2026-11-{r.randint(1, 28):02d}T{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:00Z host-{r.randint(1, 500)} {m[0]}.{m[1]} status={r.choice([200, 401, 403, 500])} {m[2]}={r.randint(0, 99999)} {m[3]} {m[4]}"
        else:
            l = f"- Constat {i} : {m[0]} {m[1]} sur {m[2]}, impact {m[3]}, preuve {m[4]}-{r.randint(100, 999)}, statut {m[5]}."
        lignes.append(l); n += len(l) + 1
    return "\n".join(lignes)


def texte(nature, n_jetons, graine):
    """Texte fabriqué d'environ n_jetons : 'code', 'scan', 'journal' ou 'notes'."""
    ratio = CAR_PAR_JETON.get(nature, 3.0)
    t = _fabriquer(nature, n_jetons * ratio, graine)
    tok = tokenizer()
    if tok is not None and n_jetons >= 50:
        reel = len(tok.encode(t).ids)
        if reel and abs(reel - n_jetons) / n_jetons > 0.02:
            t = _fabriquer(nature, n_jetons * ratio * n_jetons / reel, graine)
    return t


def definitions_outils(noms, n_jetons_par_outil, graine):
    """Définitions d'outils au format OpenAI, avec une description qui donne la taille visée."""
    outils = []
    for k, nom in enumerate(noms):
        outils.append({"type": "function", "function": {
            "name": nom,
            "description": texte("notes", n_jetons_par_outil, graine + k),
            "parameters": {"type": "object",
                           "properties": {"cible": {"type": "string"}, "detail": {"type": "string"}},
                           "required": ["cible"]}}})
    return outils


# ---------------------------------------------------------------- appel mesuré
class Modele:
    """Client OpenAI vers le serveur testé ; chaque appel est chronométré et journalisé."""

    def __init__(self, url, modele, journal, mode, delai_max_s=1800, cle_api="banc"):
        # max_retries=0 : une erreur doit être journalisée, pas masquée par une nouvelle tentative du client.
        self.client = OpenAI(base_url=url.rstrip("/") + "/v1", api_key=cle_api, timeout=delai_max_s, max_retries=0)
        self.modele, self.journal, self.mode = modele, journal, mode

    def appeler(self, messages, n_sortie, ctx, outils=None):
        """Un appel en streaming. ctx = dict (agent, tache, fil, etape) recopié dans le journal.
        Mode 'scripte' : sortie forcée à n_sortie jetons (min_tokens + ignore_eos, paramètres vLLM),
        sans appel d'outil. Mode 'autonome' : outils passés au modèle, longueur libre (plafond n_sortie)."""
        args = dict(model=self.modele, messages=messages, max_tokens=n_sortie, stream=True,
                    stream_options={"include_usage": True}, temperature=0.7)
        if self.mode == "scripte":
            args["extra_body"] = {"min_tokens": n_sortie, "ignore_eos": True}
        elif outils:
            args["tools"] = outils; args["tool_choice"] = "auto"
        t0 = time.time(); t1 = None; texte_sortie = []; appels_outils = {}; usage = None; morceaux = 0
        erreur = None
        try:
            for ev in self.client.chat.completions.create(**args):
                if ev.usage is not None:
                    usage = ev.usage
                if not ev.choices:
                    continue
                d = ev.choices[0].delta
                # Raisonnement (parseur de raisonnement de vLLM) : champ « reasoning » ou « reasoning_content ».
                extra = getattr(d, "model_extra", None) or {}
                raisonnement = extra.get("reasoning") or extra.get("reasoning_content") \
                    or getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None)
                if (d.content or d.tool_calls or raisonnement) and t1 is None:
                    t1 = time.time()
                if raisonnement:
                    morceaux += 1
                if d.content:
                    texte_sortie.append(d.content); morceaux += 1
                for tc in d.tool_calls or []:
                    a = appels_outils.setdefault(tc.index, {"nom": "", "args": ""})
                    if tc.function and tc.function.name: a["nom"] += tc.function.name
                    if tc.function and tc.function.arguments: a["args"] += tc.function.arguments
                    morceaux += 1
        except Exception as e:  # noqa: BLE001 — l'erreur est journalisée, la tâche continue ou s'arrête
            erreur = f"{type(e).__name__}: {e}"[:300]
        t2 = time.time()
        n_out = getattr(usage, "completion_tokens", None)
        n_in = getattr(usage, "prompt_tokens", None)
        det = getattr(usage, "prompt_tokens_details", None)
        n_cache = getattr(det, "cached_tokens", None) if det is not None else None
        vitesse = None
        if t1 is not None and n_out and n_out > 1 and t2 > t1:
            vitesse = (n_out - 1) / (t2 - t1)
        self.journal.appel(**ctx, t_debut=t0, ttft_s=(t1 - t0) if t1 else None, duree_s=t2 - t0,
                           jetons_entree=n_in, jetons_sortie=n_out, jetons_cache=n_cache,
                           morceaux=morceaux, appels_outils=len(appels_outils), vitesse=vitesse, erreur=erreur)
        if erreur:
            raise RuntimeError(erreur)
        return "".join(texte_sortie), list(appels_outils.values())
