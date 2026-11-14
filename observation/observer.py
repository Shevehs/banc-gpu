"""Mode observation : mesure l'usage réel, sans générer de charge.

Tes agents (ou tes utilisateurs) appellent vLLM directement ou via LiteLLM ; ce script relève /metrics
de chaque serveur vLLM au début et à la fin de chaque fenêtre (et les jauges toutes les --pas secondes),
puis enregistre les variations : jetons, requêtes, préemptions, cache, MTP, histogrammes du TTFT et
du temps par jeton de chaque requête.

  --duree      : durée totale de l'observation (s)
  --intervalle : découpe l'observation en fenêtres successives de cette durée (vue par période) ;
                 sans --intervalle, une seule fenêtre
  --litellm    : URL /metrics de LiteLLM (callback « prometheus ») pour la vue par agent ;
                 --par : étiquette qui distingue les agents (api_key_alias par défaut, ou metadata_<champ>)

Exemple (une journée en fenêtres de 15 min, avec la vue par agent) :
  python observation/observer.py --metriques "http://10.0.0.5:8000/metrics" --duree 86400 --intervalle 900 \
      --litellm http://10.0.0.9:4000/metrics/ --etiquette prod-r1 --gpu 8 --cible 30
Résultats : <sortie>/<etiquette>/observation-<AAAAMMJJ-HHMMSS>.json (une par fenêtre) et config_serveur.json.
--url et --modele (optionnels) : relève aussi l'empreinte (parallélisme) par une complétion d'un jeton,
envoyée AVANT la première fenêtre. Clé : BANC_CLE_API ou CLE_API pour la cible (avec --url) ;
BANC_CLE_LITELLM (sinon la même) pour le /metrics/ de LiteLLM, protégé par défaut ; jamais de clé vers vLLM /metrics.
Pour une interactivité exacte, lancer vLLM avec --custom-histogram-buckets (voir le README).
Bibliothèque standard seulement."""
import argparse, json, os, re, shutil, signal, sys, time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI); sys.path.insert(0, os.path.join(ICI, "..", "outils"))
from analyse import (COMPTEURS, HISTOS, delta_litellm, indicateurs, lire, resumer,  # noqa: E402
                     resumer_litellm, verdict)

NOM_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
PAR_OK = re.compile(r"^(api_key_alias|team_alias|user|end_user|metadata_[A-Za-z0-9_]{1,40})$")


class Arret(BaseException):
    """Arrêt demandé (SIGTERM) : BaseException, pour ne pas être avalé par un `except Exception`."""


ZONE, DEMANDE = [False], [False]  # lecture finale en cours ; arrêt demandé pendant celle-ci


def instantane(urls):
    """Somme des séries de toutes les URL vLLM, plus un témoin par URL (jetons de sortie, requêtes terminées)
    pour repérer un redémarrage ; lève une exception si une URL ne répond pas."""
    tout, temoins = [], {}
    for u in urls:
        series = lire(u)
        c, _, _ = resumer(series)
        temoins[u] = (c.get(COMPTEURS["jetons_sortie"], 0.0), c.get(COMPTEURS["requetes_terminees"], 0.0))
        tout += series
    return resumer(tout) + (temoins,)


def instantane_tenace(urls, essais=4, pause_s=3.0):
    """instantane() avec quelques nouvelles tentatives : une panne de quelques secondes ne perd pas la fenêtre."""
    for i in range(essais):
        try:
            return instantane(urls)
        except Exception:  # noqa: BLE001
            if i == essais - 1:
                raise
            time.sleep(pause_s)


def baisse(avant, apres):
    """Vrai si un témoin a baissé sur une URL (serveur redémarré)."""
    return any(u in avant and (apres[u][0] < avant[u][0] or apres[u][1] < avant[u][1]) for u in apres)


def fenetre(p, urls, cle, debut, fin_prevue, litellm0):
    """Observe une fenêtre ; renvoie (obs, résumé LiteLLM de fin)."""
    c0, h0, _, t_prec = instantane_tenace(urls)
    kv_max = attente_max = en_cours_max = None; lectures = echecs = 0; arret = False; redemarrage = False
    try:
        while time.time() < fin_prevue:
            time.sleep(max(0.0, min(p.pas, fin_prevue - time.time())))
            try:
                _, _, j, t = instantane(urls)
                lectures += 1
                redemarrage |= baisse(t_prec, t); t_prec = t
                if j["kv"]:
                    kv_max = max(j["kv"]) if kv_max is None else max(kv_max, max(j["kv"]))
                if j["attente"]:
                    v = sum(j["attente"]); attente_max = v if attente_max is None else max(attente_max, v)
                if j["en_cours"]:
                    v = sum(j["en_cours"]); en_cours_max = v if en_cours_max is None else max(en_cours_max, v)
            except Exception:  # noqa: BLE001 — une lecture manquée n'arrête pas la fenêtre
                echecs += 1
    except (KeyboardInterrupt, Arret):
        arret = True
    ZONE[0] = True  # SIGTERM pendant les lectures finales (vLLM puis LiteLLM) : noté, appliqué après
    try:
        c1, h1, _, t1 = instantane_tenace(urls)
        litellm_fin, litellm_err = None, None
        if p.litellm:
            try:
                litellm_fin = resumer_litellm(lire(p.litellm, prefixe="litellm_", cle=cle), p.par)
            except Exception as e:  # noqa: BLE001
                litellm_err = f"{type(e).__name__}: {e}"[:200]
    finally:
        ZONE[0] = False
    fin = time.time()
    compteurs, remise_a_zero = {}, redemarrage or baisse(t_prec, t1)
    for cle_c, nom in COMPTEURS.items():
        if nom in c0 and nom in c1:
            compteurs[cle_c] = c1[nom] - c0[nom]; remise_a_zero |= compteurs[cle_c] < 0
        elif nom in c1 and cle_c == "requetes_en_erreur":
            compteurs[cle_c] = c1[nom]  # séries apparues pendant la fenêtre : parties de 0
        else:
            compteurs[cle_c] = None
    histos = {}
    for k in HISTOS:
        histos[k] = {le: h1[k][le] - h0[k].get(le, 0.0) for le in h1[k]}
        remise_a_zero |= any(v < 0 for v in histos[k].values())
    obs = {"etiquette": p.etiquette, "niveau": p.niveau, "debut": debut, "fin": fin, "urls": urls,
           "compteurs": compteurs, "histogrammes": histos,
           "jauges": {"kv_max": kv_max, "attente_max": attente_max, "en_cours_max": en_cours_max},
           "lectures": lectures, "lectures_echouees": echecs, "remise_a_zero": remise_a_zero, "interrompue": arret}
    arret = arret or DEMANDE[0]
    obs["interrompue"] = arret
    litellm1 = litellm_fin
    if p.litellm:
        if litellm_err:
            obs["agents_erreur"] = litellm_err
        elif litellm0 is not None:
            try:
                obs["agents"] = delta_litellm(litellm0, litellm1); obs["agents_par"] = p.par
            except Exception as e:  # noqa: BLE001
                obs["agents_erreur"] = f"{type(e).__name__}: {e}"[:200]
    return obs, litellm1, arret


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--metriques", required=True, help="URL /metrics de vLLM, séparées par des espaces")
    a.add_argument("--duree", type=float, default=900, help="secondes d'observation en tout")
    a.add_argument("--intervalle", type=float, default=None, help="secondes par fenêtre (vue par période)")
    a.add_argument("--etiquette", required=True)
    a.add_argument("--niveau", type=int, default=None, help="nombre d'agents ou de sessions actifs (noté à la main)")
    a.add_argument("--litellm", default=None, help="URL /metrics de LiteLLM, pour la vue par agent")
    a.add_argument("--par", default="api_key_alias", help="étiquette LiteLLM qui distingue les agents")
    a.add_argument("--sortie", default="resultats-observation")
    a.add_argument("--ecraser", action="store_true")
    a.add_argument("--pas", type=float, default=5.0, help="secondes entre deux lectures des jauges")
    a.add_argument("--url", default=None, help="cible (vLLM ou LiteLLM) pour la configuration ; défaut : 1re URL de métriques")
    a.add_argument("--modele", default=None, help="si donné, complétion d'un jeton avant la première fenêtre (empreinte)")
    a.add_argument("--cible", type=float, default=None, help="pour le résumé affiché après chaque fenêtre")
    a.add_argument("--gpu", type=int, default=None, help="pour le résumé affiché après chaque fenêtre")
    p = a.parse_args()
    if not NOM_OK.match(p.etiquette) or ".." in p.etiquette:
        raise SystemExit("--etiquette : lettres, chiffres, point, tiret, souligné ; ni / ni ..")
    if not PAR_OK.match(p.par):
        raise SystemExit("--par : api_key_alias, team_alias, user, end_user ou metadata_<champ>")
    intervalle = p.intervalle or p.duree
    if intervalle < 60 or p.duree < intervalle:
        raise SystemExit("--intervalle (ou --duree) : au moins 60 s, et --duree ≥ --intervalle")
    urls = p.metriques.split()
    dossier = os.path.join(p.sortie, p.etiquette)
    if os.path.isdir(dossier) and os.listdir(dossier):
        if not p.ecraser:
            raise SystemExit(f"{dossier} existe déjà : change d'étiquette, ou ajoute --ecraser.")
        shutil.rmtree(dossier)
    os.makedirs(dossier, exist_ok=True)
    cle = os.environ.get("BANC_CLE_API") or os.environ.get("CLE_API") or None
    # Clé du /metrics/ de LiteLLM : BANC_CLE_LITELLM si elle existe (cible = vLLM), sinon la clé de la cible.
    cle_ll = os.environ.get("BANC_CLE_LITELLM") or cle
    try:
        from config_serveur import releve, resume
        base = (p.url or urls[0]).rstrip("/")
        base = base[: -len("/metrics")] if base.endswith("/metrics") else base
        cfg = releve(base, urls, cle if p.url else None, p.modele)  # sans --url : pas de clé vers vLLM
        with open(os.path.join(dossier, "config_serveur.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1, ensure_ascii=False)
        print(resume(cfg), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(configuration du serveur non relevée : {e})", file=sys.stderr)

    def _stop(*_):
        if ZONE[0]:
            DEMANDE[0] = True
            return
        raise Arret()
    signal.signal(signal.SIGTERM, _stop)  # l'arrêt depuis l'interface termine la fenêtre en cours proprement
    litellm = None
    if p.litellm:
        try:
            litellm = resumer_litellm(lire(p.litellm, prefixe="litellm_", cle=cle_ll), p.par)
            print(f"[litellm] {len(litellm)} valeur(s) de {p.par} vues dans {p.litellm}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[litellm] lecture impossible ({type(e).__name__}: {e}) : pas de vue par agent", flush=True)
    t_fin = time.time() + p.duree
    print(f"== {p.etiquette} : observation de {p.duree:.0f} s en fenêtre(s) de {intervalle:.0f} s"
          + (f", niveau {p.niveau}" if p.niveau is not None else "") + " (Ctrl-C pour arrêter) ==", flush=True)
    n_fen = 0
    while True:
        if n_fen and t_fin - time.time() < min(60.0, intervalle) / 2:  # pas de fenêtre résiduelle trop courte
            break
        n_fen += 1
        debut = time.time()
        fin_prevue = min(debut + intervalle, t_fin)
        try:
            obs, litellm_fin, arret = fenetre(p, urls, cle_ll, debut, fin_prevue, litellm)
        except Arret:
            print("Arrêt demandé.", flush=True)
            break
        except Exception as e:  # noqa: BLE001 — /metrics injoignable au début ou à la fin : fenêtre incomplète
            obs = {"etiquette": p.etiquette, "niveau": p.niveau, "debut": debut, "fin": time.time(), "urls": urls,
                   "incomplete": True, "erreur": f"{type(e).__name__}: {e}"[:200]}
            print(f"   fenêtre {time.strftime('%H:%M:%S', time.localtime(debut))} incomplète : /metrics injoignable "
                  f"({obs['erreur']}) ; on continue.", flush=True)
            with open(os.path.join(dossier, "observation-" + time.strftime("%Y%m%d-%H%M%S", time.localtime(debut))
                                   + ".json"), "w", encoding="utf-8") as fh:
                json.dump(obs, fh, indent=1)
            try:
                time.sleep(max(0.0, fin_prevue - time.time()))
            except (KeyboardInterrupt, Arret):
                break
            litellm = None  # point de départ LiteLLM perdu : la vue par agent reprend à la fenêtre suivante
            continue
        litellm = litellm_fin if litellm_fin is not None else litellm
        nom = "observation-" + time.strftime("%Y%m%d-%H%M%S", time.localtime(debut)) + ".json"
        with open(os.path.join(dossier, nom), "w", encoding="utf-8") as fh:
            json.dump(obs, fh, indent=1)
        msg = f"   fenêtre {time.strftime('%H:%M:%S', time.localtime(debut))} ({obs['fin'] - debut:.0f} s) -> {nom}"
        if p.cible and p.gpu and obs["remise_a_zero"]:
            msg += " ; écartée (serveur redémarré)"
        elif p.cible and p.gpu:
            r = indicateurs(obs, p.cible, p.gpu)
            f = lambda x, n=1: "—" if x is None else f"{x:.{n}f}"
            part = r["part_requetes_a_la_cible"]
            msg += (f" ; requêtes {f(r['requetes_terminees'], 0)} ; débit/GPU {f(r['debit_sortie_par_gpu'])} ; "
                    f"à {p.cible:g} jetons/s {f(None if part is None else 100 * part, 0)} %"
                    f"{'' if r['part_exacte'] is not False else ' (interpolée)'} ; {verdict(r['tient'])}")
        print(msg, flush=True)
        if obs["remise_a_zero"]:
            print("   ATTENTION : compteurs en baisse (serveur redémarré pendant la fenêtre ?) : fenêtre inutilisable.")
        if arret:
            print("Arrêt demandé : dernière fenêtre raccourcie.", flush=True)
            break
    print(f"Terminé -> {dossier}", flush=True)


if __name__ == "__main__":
    main()
