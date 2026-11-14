"""Interface du banc : petit serveur local qui lance les scripts du banc et lit leurs résultats.

Usage (depuis la racine du dépôt, environnement .venv-aiperf activé) :
    python interface/serveur.py            # puis ouvrir http://127.0.0.1:8765
    python interface/serveur.py --port 9000
Sur une machine distante : tunnel SSH depuis ton poste, ex. ssh -L 8765:127.0.0.1:8765 <machine>.

Le serveur n'écoute que sur 127.0.0.1 et exige un jeton aléatoire, affiché au démarrage dans l'adresse à
ouvrir : une autre page web ou un autre utilisateur de la machine ne peut pas lancer d'essai sans lui.
Il vérifie aussi l'en-tête Host et l'origine des requêtes. Il n'exécute que balayage.sh, agentx.sh,
agents_test/lanceur.py et les scripts de dépouillement (dont observation/interroger.py, qui interroge Prometheus),
avec des paramètres vérifiés (options AIPerf sur liste blanche) ; un essai lancé ici est identique au même essai
lancé à la main (la commande exacte est écrite dans son journal, sans la clé d'API). Un seul essai à la fois ;
l'observation n'est pas un essai : elle lit Prometheus et peut s'afficher pendant qu'un essai tourne.
Dépendance : aiohttp (déjà installé avec AIPerf).
"""
import argparse, asyncio, csv, json, os, re, secrets, shlex, signal, sys, time
from pathlib import Path

from aiohttp import web

ICI = Path(__file__).resolve().parent
BANC = ICI.parent
DIST = ICI / "dist"
JOURNAUX = ICI / "journaux"
NOM_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
JETON = os.environ.get("BANC_JETON") or secrets.token_urlsafe(24)
PORT = [8765]
# Options AIPerf permises dans « options en plus » : nom -> nombre de valeurs attendues
EXTRA_OK = {"--use-server-token-count": 0, "--benchmark-grace-period": 1, "--prompt-corpus": 1,
            "--num-dataset-entries": 1, "--request-timeout-seconds": 1}
PALIERS_OK = re.compile(r"^\s*\d{1,4}(\s+\d{1,4}){0,30}\s*$")
ESSAIS = {}  # id -> dict(type, commande, debut, fin, code, journal, proc)
COMPTEUR = [0]


class Refus(Exception):
    pass


MODELE_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,199}$")


def modele(v):
    """Nom servi ou alias de passerelle (peut contenir / et :, ex. Qwen/Qwen3-0.6B)."""
    v = str(v or "").strip()
    if not MODELE_OK.match(v) or ".." in v:
        raise Refus("modèle : lettres, chiffres et . _ - / : @, en commençant par une lettre ou un chiffre")
    return v


def nom(v, champ):
    v = str(v or "").strip()
    if not NOM_OK.match(v) or ".." in v:
        raise Refus(f"{champ} : lettres, chiffres, point, tiret ou souligné, en commençant par une lettre ou un chiffre")
    return v


def extra_ok(v):
    """Options AIPerf supplémentaires, limitées à une liste blanche."""
    mots = shlex.split(str(v or ""))
    i = 0
    while i < len(mots):
        m = mots[i]
        if m not in EXTRA_OK:
            raise Refus(f"option AIPerf non permise : {m} (permises : {', '.join(EXTRA_OK)})")
        n = EXTRA_OK[m]
        vals = mots[i + 1:i + 1 + n]
        if len(vals) < n or any(not re.match(r"^[A-Za-z0-9._-]{1,40}$", x) for x in vals):
            raise Refus(f"valeur invalide pour {m}")
        i += 1 + n
    return " ".join(mots)


def nombre(v, champ, mini, maxi, entier=False):
    try:
        x = int(v) if entier else float(v)
    except (TypeError, ValueError):
        raise Refus(f"{champ} : nombre attendu")
    if not mini <= x <= maxi:
        raise Refus(f"{champ} : entre {mini} et {maxi}")
    return x


def url_locale(v):
    v = str(v or "").strip()
    if not re.match(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?/?$", v):
        raise Refus("URL de la cible : forme http://hôte:port attendue")
    return v.rstrip("/")


def urls_metriques(v):
    """URL /metrics de vLLM, séparées par des espaces (optionnel)."""
    urls = str(v or "").split()
    for u in urls:
        if not re.match(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?(/[A-Za-z0-9._/-]*)?$", u):
            raise Refus(f"URL des métriques : forme http://hôte:port/metrics attendue ({u})")
    return urls


def cle(v):
    v = str(v or "").strip()
    if v and not re.match(r"^[\x21-\x7e]{1,500}$", v):
        raise Refus("clé d'API : caractères non permis")
    return v


SECRETS = ("CLE_API", "BANC_CLE_API", "BANC_CLES_AGENTS", "BANC_CLE_PROMETHEUS")


def dossier_resultats(v):
    """Dossier de résultats relatif à la racine du dépôt, sans remontée."""
    v = str(v or "").strip()
    if not v or not re.match(r"^[A-Za-z0-9._/-]{1,120}$", v) or ".." in v.split("/") or v.startswith("/"):
        raise Refus("dossier : chemin relatif au dépôt attendu")
    return v


def commande_synthetique(p):
    env = {
        "URL": url_locale(p.get("url")), "MODELE": modele(p.get("modele")),
        "ISL": str(nombre(p.get("isl"), "jetons d'entrée", 1, 1_000_000, True)),
        "OSL": str(nombre(p.get("osl"), "jetons de sortie", 1, 100_000, True)),
        "TEMPERATURE": str(nombre(p.get("temperature", 0.7), "température", 0, 2)),
        "RACINE": dossier_resultats(p.get("racine") or "resultats"),
    }
    tok = str(p.get("tokenizer") or "").strip()
    if tok:
        if tok.startswith("-") or not re.match(r"^[A-Za-z0-9._/@:~-]{1,300}$", tok):
            raise Refus("tokenizer : nom Hugging Face ou chemin local attendu")
        env["TOKENIZER"] = os.path.expanduser(tok)
    env.setdefault("TOKENIZER", "")  # vide = tokenizer par défaut du dépôt
    # Toutes les variables lues par balayage.sh sont fixées ici : rien n'est hérité du terminal.
    env["METRIQUES"] = " ".join(urls_metriques(p.get("metriques")))
    env["CLE_API"] = cle(p.get("cle_api")) or ""
    if p.get("duree"):
        env["DUREE"] = str(nombre(p.get("duree"), "durée", 10, 7200, True)); env["REQ_PAR_CONC"] = "8"
    else:
        env["DUREE"] = ""
        env["REQ_PAR_CONC"] = str(nombre(p.get("req_par_conc", 8), "requêtes par concurrence", 1, 1000, True))
    env["EXTRA"] = extra_ok(p.get("extra")) or ""
    env.update(ECRASER="", SEED="")
    paliers = str(p.get("paliers") or "")
    if not PALIERS_OK.match(paliers):
        raise Refus("paliers : entiers séparés par des espaces")
    cmd = ["bash", str(BANC / "balayage.sh"), nom(p.get("etiquette"), "étiquette"), " ".join(paliers.split())]
    return cmd, env, BANC


def chemin_local(v, champ):
    """Chemin d'un dossier existant sur la machine de l'interface (absolu, ~ ou relatif au dépôt)."""
    v = str(v or "").strip()
    if not re.match(r"^[A-Za-z0-9._/~-]{1,300}$", v) or v.startswith("-"):
        raise Refus(f"{champ} : chemin de dossier attendu")
    c = Path(os.path.expanduser(v))
    c = c if c.is_absolute() else BANC / c
    if not c.is_dir():
        raise Refus(f"{champ} : dossier introuvable ({c})")
    return str(c)


def commande_agentx(p):
    """Stress test AgentX (agentx.sh) ; toutes ses variables sont fixées ici."""
    court = bool(p.get("essai_court"))
    duree = nombre(p.get("duree", 1800), "durée par palier", 30 if court else 900, 7200, True)
    env = {"URL": url_locale(p.get("url")), "MODELE": modele(p.get("modele")),
           "METRIQUES": " ".join(urls_metriques(p.get("metriques"))), "CLE_API": cle(p.get("cle_api")) or "",
           "DUREE": str(duree), "RACINE": dossier_resultats(p.get("racine") or "resultats-agentx"),
           "AGENTX_DOSSIER": chemin_local(p.get("dossier_traces"), "dossier des traces") if p.get("dossier_traces") else "",
           "ESSAI_COURT": "1" if court else "", "ECRASER": "", "SEED": "", "TOKENIZER": "",
           "CONTEXTE_MAX": str(nombre(p.get("contexte_max"), "contexte max", 1024, 10_000_000, True)) if p.get("contexte_max") else "",
           "AGENTX_VARIANTE": "256k" if p.get("variante") == "256k" else ""}
    tok = str(p.get("tokenizer") or "").strip()
    if tok:
        if tok.startswith("-") or not re.match(r"^[A-Za-z0-9._/@:~-]{1,300}$", tok):
            raise Refus("tokenizer : nom Hugging Face ou chemin local attendu")
        env["TOKENIZER"] = os.path.expanduser(tok)
    paliers = str(p.get("paliers") or "")
    if not PALIERS_OK.match(paliers):
        raise Refus("paliers : entiers séparés par des espaces")
    cmd = ["bash", str(BANC / "agentx.sh"), nom(p.get("etiquette"), "étiquette"), " ".join(paliers.split())]
    return cmd, env, BANC


def commande_agents(p):
    agents = [a for a in (p.get("agents") or []) if a in ("revue_code", "pentest", "rapport")]
    if not agents:
        raise Refus("agents : au moins un parmi revue_code, pentest, rapport")
    paliers = str(p.get("paliers") or "")
    if not PALIERS_OK.match(paliers):
        raise Refus("paliers : entiers séparés par des espaces")
    mode = p.get("mode") if p.get("mode") in ("scripte", "autonome") else None
    if not mode:
        raise Refus("mode : scripte ou autonome")
    env = {"BANC_CLE_API": cle(p.get("cle_api"))} if cle(p.get("cle_api")) else {}
    paires = []
    for paire in str(p.get("cles_agents") or "").split(","):
        if paire.strip():
            k_, _, v_ = paire.partition("=")
            if k_.strip() not in ("revue_code", "pentest", "rapport") or not cle(v_):
                raise Refus("clés par agent : revue_code=<clé>,pentest=<clé>,rapport=<clé>")
            paires.append(f"{k_.strip()}={cle(v_)}")
    if paires:
        env["BANC_CLES_AGENTS"] = ",".join(paires)
    met = urls_metriques(p.get("metriques"))
    cmd = [sys.executable, str(BANC / "agents_test" / "lanceur.py"),
           "--url", url_locale(p.get("url")), "--modele", modele(p.get("modele")),
           "--agents", ",".join(agents), "--paliers", " ".join(paliers.split()),
           "--duree", str(nombre(p.get("duree"), "durée", 10, 7200)),
           "--prechauffe", str(nombre(p.get("prechauffe", 60), "préchauffe", 0, 3600)),
           "--echelle", str(nombre(p.get("echelle", 1), "échelle", 0.01, 2)),
           "--facteur-latence", str(nombre(p.get("facteur_latence", 1), "facteur de latence", 0, 10)),
           "--mode", mode, "--etiquette", nom(p.get("etiquette"), "étiquette"),
           "--sortie", str(BANC / dossier_resultats(p.get("racine") or "agents_test/resultats-agents"))]
    if p.get("etiqueter_agents"):
        cmd.append("--etiqueter-agents")
    if met:
        cmd += ["--metriques", " ".join(met)]
    return cmd, env, BANC / "agents_test"


# Seules ces variables du terminal passent aux essais ; les réglages du banc viennent des champs de l'interface.
ENV_HERITE = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TERM", "VIRTUAL_ENV", "HF_HOME",
              "HF_HUB_CACHE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
              "https_proxy", "http_proxy", "no_proxy", "DO_NOT_TRACK", "TZ"}


def environnement_de_base():
    return {k: v for k, v in os.environ.items() if k in ENV_HERITE or k.startswith("LC_")}


async def suivre(eid, proc, f):
    code = await proc.wait()
    f.write(f"\n== fin, code de sortie {code} ==\n"); f.close()
    ESSAIS[eid].update(fin=time.time(), code=code)


async def corps(req):
    try:
        p = await req.json()
    except Exception:  # noqa: BLE001
        raise Refus("corps JSON attendu")
    if not isinstance(p, dict):
        raise Refus("corps JSON attendu")
    return p


async def lancer(req):
    if any(e["code"] is None for e in ESSAIS.values()):
        return web.json_response({"erreur": "un essai est déjà en cours : attends sa fin ou arrête-le"}, status=409)
    try:
        p = await corps(req)
    except Refus as e:
        return web.json_response({"erreur": str(e)}, status=400)
    try:
        fab = {"synthetique": commande_synthetique, "agents": commande_agents,
               "agentx": commande_agentx}.get(p.get("type"))
        if not fab:
            raise Refus("type d'essai inconnu")
        cmd, env, cwd = fab(p)
    except Refus as e:
        return web.json_response({"erreur": str(e)}, status=400)
    JOURNAUX.mkdir(exist_ok=True)
    COMPTEUR[0] += 1
    eid = f"{time.strftime('%Y%m%d-%H%M%S')}-{COMPTEUR[0]:03d}-{p.get('etiquette', '')}"
    journal = JOURNAUX / f"{eid}.log"
    f = open(journal, "w", encoding="utf-8")
    affiche = " ".join(f"{k}={'***' if k in SECRETS else repr(v) if ' ' in v else v}" for k, v in env.items()) \
        + (" " if env else "") + " ".join(f"'{c}'" if " " in c else c for c in cmd)
    f.write(f"$ cd {cwd}\n$ {affiche}\n\n"); f.flush()
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), env={**environnement_de_base(), **env, "PYTHONUNBUFFERED": "1"},
                                                stdout=f, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    ESSAIS[eid] = dict(id=eid, type=p.get("type"), etiquette=p.get("etiquette"), commande=affiche,
                       debut=time.time(), fin=None, code=None, journal=str(journal), proc=proc)
    asyncio.create_task(suivre(eid, proc, f))
    return web.json_response({"id": eid})


def public(e):
    return {k: v for k, v in e.items() if k != "proc"}


async def liste(req):
    return web.json_response(sorted((public(e) for e in ESSAIS.values()), key=lambda e: -e["debut"]))


async def journal(req):
    e = ESSAIS.get(req.match_info["eid"])
    if not e:
        return web.json_response({"erreur": "essai inconnu"}, status=404)
    try:
        depuis = max(0, int(req.query.get("depuis", "0")))
    except ValueError:
        depuis = 0
    with open(e["journal"], "rb") as fh:
        fh.seek(depuis); morceau = fh.read(200_000)
    return web.json_response({"texte": morceau.decode("utf-8", "replace"), "position": depuis + len(morceau),
                              "termine": e["code"] is not None})


async def arreter(req):
    e = ESSAIS.get(req.match_info["eid"])
    if not e or e["code"] is not None:
        return web.json_response({"erreur": "essai inconnu ou terminé"}, status=400)
    try:
        os.killpg(e["proc"].pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return web.json_response({"ok": True})


async def racines(req):
    """Dossiers de résultats présents, par type (d'après leur contenu)."""
    synth, agents, agentx = [], [], []
    for d in sorted(BANC.glob("resultats*")) + sorted((BANC / "agents_test").glob("resultats*")):
        if d.is_dir():
            rel = str(d.relative_to(BANC))
            if any(d.glob("*/c*/profile_export_aiperf.json")):
                (agentx if any(d.glob("*/agentx.txt")) else synth).append(rel)
            if any(d.glob("*/n*/palier.json")):
                agents.append(rel)
    return web.json_response({"synthetique": synth, "agents": agents, "agentx": agentx})


async def resultats(req):
    q = req.query
    try:
        racine = dossier_resultats(q.get("racine"))
        cible = nombre(q.get("cible"), "cible", 0.1, 10_000)
        gpu = nombre(q.get("gpu"), "GPU", 1, 4096, True) if q.get("gpu") not in (None, "") else None
    except Refus as e:
        return web.json_response({"erreur": str(e)}, status=400)
    typ = q.get("type")
    script, tableau = {"agents": ("agents_test/depouiller_agents.py", "tableau_agents.csv")
                       }.get(typ, ("depouiller.py", "tableau.csv"))
    cmd = [sys.executable, str(BANC / script), str(BANC / racine), "--cible", str(cible)] + (["--gpu", str(gpu)] if gpu else [])
    if q.get("exiger") == "1":
        cmd.append("--exiger-metriques")
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    sortie = (await proc.communicate())[0].decode("utf-8", "replace")
    fichier = BANC / racine / tableau
    lignes = []
    if proc.returncode == 0 and fichier.exists():
        with open(fichier, newline="", encoding="utf-8") as fh:
            lignes = list(csv.DictReader(fh))
    messages = [l for l in sortie.splitlines() if "retenu" in l or "ATTENTION" in l or "aucun" in l.lower()
                or l.startswith("GPU :") or l.startswith("ERREUR")
                or "qui tient" in l]
    return web.json_response({"lignes": lignes, "messages": messages, "sortie": sortie, "code": proc.returncode,
                              "commande": " ".join(cmd)})


SEL_OK = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*(=|!=|=~|!~)\s*"[^"\\]*"'
                    r'(\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*(=|!=|=~|!~)\s*"[^"\\]*")*\s*$')
DUREE_OK = re.compile(r"^\d{1,6}[smhd]?$")
DATE_OK = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$")


async def observation(req):
    """Observation : interroge Prometheus (observation/interroger.py) ; rien à lancer, aucune charge générée.
    Les plages des essais lancés depuis cette interface sur la même période sont écartées (--exclure)."""
    try:
        p = await corps(req)
        u = str(p.get("prometheus") or "").strip()
        if not re.match(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?(/[A-Za-z0-9._/-]*)?$", u):
            raise Refus("URL de Prometheus : forme http://hôte:9090 attendue")
        cmd = [sys.executable, str(BANC / "observation" / "interroger.py"), "--prometheus", u.rstrip("/"),
               "--cible", str(nombre(p.get("cible"), "cible", 0.1, 10_000)),
               "--gpu", str(nombre(p.get("gpu"), "GPU", 1, 4096, True))]
        sel = str(p.get("vllm") or 'job="vllm"')
        if not SEL_OK.match(sel):
            raise Refus('sélecteur vLLM : ex. job="vllm"')
        cmd += ["--vllm", sel]
        vue = p.get("vue") if p.get("vue") in ("fenetres", "periode", "agents") else "fenetres"
        if p.get("litellm"):
            sl = str(p.get("litellm"))
            par = str(p.get("par") or "metadata_agent").strip()
            if not SEL_OK.match(sl):
                raise Refus('sélecteur LiteLLM : ex. job="litellm"')
            if not re.match(r"^(api_key_alias|team_alias|user|end_user|metadata_[A-Za-z0-9_]{1,40})$", par):
                raise Refus("agents distingués par : api_key_alias, team_alias, user, end_user ou metadata_<champ>")
            cmd += ["--litellm", sl, "--par", par]
        elif vue == "agents":
            raise Refus("vue par agent : donne le sélecteur LiteLLM")
        for k in ("fenetre", "collecte", "depuis"):
            v = str(p.get(k) or "").strip()
            if v:
                if not DUREE_OK.match(v) or (k == "depuis" and v.endswith("d") and int(v[:-1]) > 400):
                    raise Refus(f"{k} : durée attendue, ex. 5m, 2h, 1d (400 jours au plus)")
                cmd += [f"--{k}", v]
        for k in ("debut", "fin"):
            v = str(p.get(k) or "").strip()
            if v:
                if not DATE_OK.match(v):
                    raise Refus(f"{k} : date attendue, forme AAAA-MM-JJ HH:MM")
                cmd += [f"--{k}", v.replace("T", " ")]
        if p.get("ttft_max") not in (None, ""):
            cmd += ["--ttft-max", str(nombre(p.get("ttft_max"), "TTFT max", 1, 10_000_000))]
        cmd += ["--vue", vue, "--periode", "heure" if p.get("periode") == "heure" else "plage"]
        k = cle(p.get("cle_prometheus"))
    except Refus as e:
        return web.json_response({"erreur": str(e)}, status=400)
    for e in ESSAIS.values():  # stress tests lancés d'ici : leur charge ne doit pas compter comme usage réel
        fin = e["fin"] or time.time()
        cmd += ["--exclure", f"{e['debut']:.0f}/{fin + 60:.0f}"]
    JOURNAUX.mkdir(exist_ok=True)
    csv_f = JOURNAUX / f"observation-{secrets.token_hex(4)}.csv"
    proc = await asyncio.create_subprocess_exec(*cmd, "--sortie", str(csv_f), cwd=str(BANC),
                                                env={**environnement_de_base(), **({"BANC_CLE_PROMETHEUS": k} if k else {})},
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        sortie = (await asyncio.wait_for(proc.communicate(), timeout=120))[0].decode("utf-8", "replace")
    except asyncio.TimeoutError:
        proc.kill()
        return web.json_response({"erreur": "Prometheus n'a pas répondu en 120 s : réduis la plage"}, status=504)
    lignes = []
    if csv_f.exists():
        with open(csv_f, newline="", encoding="utf-8") as fh:
            lignes = list(csv.DictReader(fh))
        csv_f.unlink()
    messages = [l for l in sortie.splitlines() if "ATTENTION" in l or "Plus haute" in l or l.startswith("Aucun")
                or "injoignable" in l or "HTTP" in l]
    return web.json_response({"lignes": lignes, "messages": messages, "sortie": sortie, "code": proc.returncode,
                              "commande": " ".join(cmd)})


async def tester(req):
    """Vérifie la cible (/v1/models) et les métriques vLLM (/metrics), et relève la configuration du serveur."""
    import aiohttp
    try:
        p = await corps(req)
        url = url_locale(p.get("url")); k = cle(p.get("cle_api"))
        met = urls_metriques(p.get("metriques")) or [url + "/metrics"]
        mod = modele(p.get("modele")) if p.get("modele") else None
    except Refus as e:
        return web.json_response({"erreur": str(e)}, status=400)
    # Pendant un essai, pas de complétion d'un jeton : elle tomberait dans la fenêtre mesurée.
    en_cours = any(e["code"] is None for e in ESSAIS.values())
    if en_cours:
        mod = None
    entetes = {"Authorization": f"Bearer {k}"} if k else {}
    res = {"essai_en_cours": en_cours, "cible": None, "metriques": [], "passerelle_sans_metriques": not urls_metriques(p.get("metriques"))}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
        try:
            async with s.get(url + "/v1/models", headers=entetes) as r:
                j = await r.json(content_type=None) if r.status == 200 else None
                res["cible"] = {"ok": r.status == 200, "statut": r.status,
                                "modeles": [m.get("id") for m in (j or {}).get("data", [])] if j else []}
        except Exception as e:  # noqa: BLE001
            res["cible"] = {"ok": False, "erreur": f"{type(e).__name__}: {e}"[:200]}
        for u in met:
            try:
                async with s.get(u, allow_redirects=False) as r:  # /metrics n'exige pas de clé : on ne l'envoie pas
                    t = await r.text() if r.status == 200 else ""
                    n = sum(1 for l in t.splitlines() if l.startswith("vllm:"))
                    res["metriques"].append({"url": u, "ok": r.status == 200 and n > 0, "statut": r.status, "lignes_vllm": n})
            except Exception as e:  # noqa: BLE001
                res["metriques"].append({"url": u, "ok": False, "erreur": f"{type(e).__name__}: {e}"[:200]})
    try:  # version, cache, parallélisme (empreinte) ; /server_info seulement si VLLM_SERVER_DEV_MODE=1
        sys.path.insert(0, str(BANC / "outils"))
        from config_serveur import releve
        res["config"] = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(
            None, releve, url, urls_metriques(p.get("metriques")), k, mod), timeout=60)
    except Exception as e:  # noqa: BLE001
        res["config"] = {"erreur": f"{type(e).__name__}: {e}"[:200]}
    return web.json_response(res)


async def accueil(req):
    if not (DIST / "index.html").exists():
        return web.Response(text="Interface non compilée : dist/ manquant (voir interface/README.md).", status=500)
    return web.FileResponse(DIST / "index.html")


@web.middleware
async def garde(req, handler):
    """Jeton, Host et origine vérifiés pour toute l'API (protège des autres pages web et du rebinding DNS)."""
    if req.path.startswith("/api/"):
        hotes = {f"127.0.0.1:{PORT[0]}", f"localhost:{PORT[0]}"}
        if req.host not in hotes:
            return web.json_response({"erreur": "hôte refusé"}, status=403)
        origine = req.headers.get("Origin")
        if origine and origine not in {f"http://{h}" for h in hotes}:
            return web.json_response({"erreur": "origine refusée"}, status=403)
        if not secrets.compare_digest(req.headers.get("X-Banc-Jeton", ""), JETON):
            return web.json_response({"erreur": "jeton absent ou invalide : ouvre l'adresse affichée au démarrage"}, status=401)
        if req.method == "POST" and req.content_type != "application/json":
            return web.json_response({"erreur": "JSON attendu"}, status=415)
    return await handler(req)


async def arret_propre(app):
    """À l'arrêt de l'interface, arrête aussi les essais en cours (sinon ils continueraient seuls)."""
    for e in ESSAIS.values():
        if e["code"] is None:
            try:
                os.killpg(e["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main():
    a = argparse.ArgumentParser(); a.add_argument("--port", type=int, default=8765)
    p = a.parse_args()
    PORT[0] = p.port
    app = web.Application(middlewares=[garde])
    app.on_shutdown.append(arret_propre)
    app.add_routes([web.post("/api/essais", lancer), web.get("/api/essais", liste),
                    web.get("/api/essais/{eid}/journal", journal), web.post("/api/essais/{eid}/arreter", arreter),
                    web.get("/api/racines", racines), web.get("/api/resultats", resultats),
                    web.post("/api/test-connexion", tester), web.post("/api/observation", observation), web.get("/", accueil)])
    if (DIST / "assets").exists():
        app.router.add_static("/assets", DIST / "assets")
    print(f"Interface du banc : http://127.0.0.1:{p.port}/?jeton={JETON}\n(ouvre cette adresse exacte ; Ctrl-C pour arrêter l'interface et les essais en cours)")
    web.run_app(app, host="127.0.0.1", port=p.port, print=None)


if __name__ == "__main__":
    main()
