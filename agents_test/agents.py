"""Trois profils d'agents de test qui reproduisent une FORME de charge agentique.

Ils ne font aucun travail offensif réel : leurs outils sont simulés (commun.texte) et leur
contenu est fabriqué. Seules comptent les tailles (prompt système, définitions d'outils,
sorties d'outils, sorties du modèle), le nombre d'étapes, le parallélisme et les temps d'outil.

Tailles par défaut (HYPOTHÈSES, à adapter à la charge visée) : revue de code ≈ 135K jetons par
sous-agent, 3 sous-agents ; pentest pas à pas ≈ 69K, 1 fil ; rapport ≈ 135K, 1 fil
(estimés avec le vocabulaire Qwen).
Toutes les tailles sont multipliées par ECHELLE (par exemple 0,2 sur un Mac).
"""
import random, threading, time

from commun import definitions_outils, texte

PROFILS = {
    "revue_code": dict(nom="Revue de code", systeme=3000, outils=4, jetons_par_outil=300,
                liste_fichiers=1500, plan=300, sous_agents=3, systeme_sous_agent=2000,
                etapes=6, lecture=22000, analyse=400, synthese=1500, latence_outil_s=5.0),
    "pentest": dict(nom="Pentest pas à pas", systeme=3000, outils=6, jetons_par_outil=300,
                etapes=15, resultat_outil=4000, reflexion=250, rapport=800, latence_outil_s=8.0),
    "rapport": dict(nom="Rédaction de rapport", systeme=2000, notes=125000, plan=800,
                 redaction=4000, latence_outil_s=2.0),
}
# Graines fixes : le prompt système et les définitions d'outils sont identiques d'une tâche à
# l'autre (comme pour un vrai agent), ce qui permet au cache de préfixes de les réutiliser.
GRAINE_FIXE = {"revue_code": 3_000_000, "pentest": 2_000_000, "rapport": 9_000_000}
OUTILS = {"revue_code": ["lister_fichiers", "lire_fichier", "chercher_motif", "analyse_statique"],
          "pentest": ["scanner_ports", "identifier_service", "lire_journal", "tester_auth", "chercher_vuln", "noter"]}
NATURE = {"lister_fichiers": "code", "lire_fichier": "code", "chercher_motif": "code", "analyse_statique": "code",
          "scanner_ports": "scan", "identifier_service": "scan", "lire_journal": "journal",
          "tester_auth": "journal", "chercher_vuln": "scan", "noter": "notes"}


# Multiplie les temps d'outil simulés (réglé par le lanceur, option --facteur-latence).
FACTEUR_LATENCE = 1.0


def _n(x, echelle, minimum=32):
    return max(minimum, int(round(x * echelle)))


def _systeme(agent, p, echelle, mode):
    """Prompt système fixe. En mode scripté, les définitions d'outils sont incluses en texte
    (même taille de prompt, sans dépendre du parseur d'appels d'outils du serveur)."""
    g = GRAINE_FIXE[agent]
    s = f"Tu es l'agent de test {p['nom']}. Réponds en texte.\n" + texte("notes", _n(p["systeme"], echelle), g)
    outils = None
    if agent in OUTILS:
        outils = definitions_outils(OUTILS[agent], _n(p["jetons_par_outil"], echelle), g + 1000)
        if mode == "scripte":
            s += "\nOutils disponibles :\n" + "\n".join(f"{o['function']['name']} : {o['function']['description']}" for o in outils)
            outils = None
    return s, outils


def _outil(nom, n_jetons, graine, latence_s):
    """Exécute un outil simulé : attend la latence prévue et renvoie un texte de taille fixée."""
    # latence variable (± 30 %) autour de la valeur prévue, pour éviter une charge trop régulière
    time.sleep(latence_s * FACTEUR_LATENCE * random.uniform(0.7, 1.3))
    return texte(NATURE.get(nom, "notes"), n_jetons, graine)


def _boucle(m, messages, outils, n_etapes, n_sortie, n_resultat, nom_outil, ctx, graine, latence_s, mode):
    """Boucle outil → modèle. Scripté : l'outil est imposé à chaque étape.
    Autonome : on exécute les outils choisis par le modèle, et on s'arrête quand il n'en appelle plus."""
    dernier = ""
    for e in range(n_etapes):
        if mode == "scripte":
            resultat = _outil(nom_outil, n_resultat, graine + e, latence_s)
            messages.append({"role": "user", "content": f"Résultat de {nom_outil} :\n{resultat}"})
            dernier, _ = m.appeler(messages, n_sortie, {**ctx, "etape": e})
            messages.append({"role": "assistant", "content": dernier})
        else:
            dernier, appels = m.appeler(messages, n_sortie, {**ctx, "etape": e}, outils)
            if not appels:
                break
            ids = [f"appel_{e}_{k}" for k in range(len(appels))]
            messages.append({"role": "assistant", "content": dernier or None, "tool_calls": [
                {"id": i, "type": "function", "function": {"name": a["nom"] or "noter", "arguments": a["args"] or "{}"}}
                for i, a in zip(ids, appels)]})
            for i, a in zip(ids, appels):
                messages.append({"role": "tool", "tool_call_id": i,
                                 "content": _outil(a["nom"], n_resultat, graine + 100 * e + len(i), latence_s)})
    return dernier


def revue_code(m, tache, graine, echelle, mode):
    p = PROFILS["revue_code"]; ctx = {"agent": "revue_code", "tache": tache}
    sysm, outils = _systeme("revue_code", p, echelle, mode)
    msgs = [{"role": "system", "content": sysm},
            {"role": "user", "content": "Revue de sécurité du dépôt. Liste des fichiers :\n"
             + _outil("lister_fichiers", _n(p["liste_fichiers"], echelle), graine, p["latence_outil_s"])}]
    plan, _ = m.appeler(msgs, _n(p["plan"], echelle), {**ctx, "fil": 0, "etape": "plan"})
    msgs.append({"role": "assistant", "content": plan})
    resumes = [None] * p["sous_agents"]

    def sous_agent(k):
        s_sys = "Sous-agent d'analyse.\n" + texte("notes", _n(p["systeme_sous_agent"], echelle), GRAINE_FIXE["revue_code"] + 7)
        s_msgs = [{"role": "system", "content": s_sys},
                  {"role": "user", "content": f"Analyse la partie {k + 1} du dépôt selon ce plan :\n{plan}"}]
        resumes[k] = _boucle(m, s_msgs, outils, p["etapes"], _n(p["analyse"], echelle), _n(p["lecture"], echelle),
                             "lire_fichier", {**ctx, "fil": k + 1}, graine * 10 + k * 1000, p["latence_outil_s"], mode)

    fils = [threading.Thread(target=sous_agent, args=(k,)) for k in range(p["sous_agents"])]
    for f in fils: f.start()
    for f in fils: f.join()
    msgs.append({"role": "user", "content": "Synthèse des sous-agents :\n" + "\n---\n".join(r or "" for r in resumes)})
    m.appeler(msgs, _n(p["synthese"], echelle), {**ctx, "fil": 0, "etape": "synthese"})


def pentest(m, tache, graine, echelle, mode):
    p = PROFILS["pentest"]; ctx = {"agent": "pentest", "tache": tache, "fil": 0}
    sysm, outils = _systeme("pentest", p, echelle, mode)
    msgs = [{"role": "system", "content": sysm},
            {"role": "user", "content": "Évalue l'environnement de test simulé, étape par étape."}]
    _boucle(m, msgs, outils, p["etapes"], _n(p["reflexion"], echelle), _n(p["resultat_outil"], echelle),
            "scanner_ports", ctx, graine, p["latence_outil_s"], mode)
    msgs.append({"role": "user", "content": "Rédige le rapport final."})
    m.appeler(msgs, _n(p["rapport"], echelle), {**ctx, "etape": "rapport"})


def rapport(m, tache, graine, echelle, mode):
    p = PROFILS["rapport"]; ctx = {"agent": "rapport", "tache": tache, "fil": 0}
    sysm, _ = _systeme("rapport", p, echelle, mode)
    time.sleep(p["latence_outil_s"] * FACTEUR_LATENCE)  # lecture des notes d'engagement (simulée)
    msgs = [{"role": "system", "content": sysm},
            {"role": "user", "content": "Notes d'engagement :\n" + texte("notes", _n(p["notes"], echelle), graine)
             + "\nPropose le plan du rapport."}]
    plan, _ = m.appeler(msgs, _n(p["plan"], echelle), {**ctx, "etape": "plan"})
    # Deuxième appel : même début que le premier, donc réutilisable par le cache de préfixes.
    msgs += [{"role": "assistant", "content": plan}, {"role": "user", "content": "Rédige le rapport complet."}]
    m.appeler(msgs, _n(p["redaction"], echelle), {**ctx, "etape": "redaction"})


AGENTS = {"revue_code": revue_code, "pentest": pentest, "rapport": rapport}
