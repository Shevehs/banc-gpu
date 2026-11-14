import React, { useEffect, useRef, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

// Jeton d'accès : il figure dans l'adresse affichée au démarrage du serveur (?jeton=...).
const JETON = (() => {
  const q = new URLSearchParams(window.location.search).get("jeton");
  try { if (q) sessionStorage.setItem("banc-jeton", q); return q || sessionStorage.getItem("banc-jeton") || ""; }
  catch { return q || ""; }
})();
const api = async (chemin, options = {}) => {
  const r = await fetch(chemin, { ...options, headers: { ...(options.headers || {}), "X-Banc-Jeton": JETON } });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.erreur || `HTTP ${r.status}`);
  return j;
};
const COULEURS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#4d7c0f", "#be185d"];
const f1 = (x, n = 1) => (x === "" || x === undefined || x === null || isNaN(+x) ? "—" : (+x).toFixed(n));
const pct = (x) => (x === "" || x === undefined || x === null || isNaN(+x) ? "—" : `${(100 * +x).toFixed(0)} %`);

// ------------------------------------------------------------------ valeurs initiales
// Aucune configuration n'est figée : tout se saisit dans le formulaire.
const CONNEXION = { url: "http://127.0.0.1:8000", modele: "banc", metriques: "", cle_api: "" };
const INITIAL = {
  synthetique: { tokenizer: "", isl: 2048, osl: 256, paliers: "1 2 4 8 16", duree: "", req_par_conc: 8,
    temperature: 0.7, racine: "resultats", extra: "", etiquette: "essai-r1" },
  agents: { agents: ["revue_code", "pentest", "rapport"], paliers: "1 2 4", duree: 300, prechauffe: 30, echelle: 0.2,
    facteur_latence: 1, mode: "scripte", racine: "agents_test/resultats-agents", etiquette: "agents-r1" },
};

function Champ({ label, aide, children }) {
  return (
    <label className="champ">
      <span className="etiquette">{label}</span>
      {children}
      {aide && <span className="aide">{aide}</span>}
    </label>
  );
}


// Configuration lue sur vLLM : /version, /v1/models, vllm:cache_config_info, empreinte system_fingerprint
// (parallélisme, mode par défaut de vLLM) et, si VLLM_SERVER_DEV_MODE=1, /server_info.
const oui = (v) => (v === true || v === "True" || v === "true" ? "activé" : v === false || v === "False" || v === "false" ? "désactivé" : v ?? "?");
function ConfigServeur({ config }) {
  if (config.erreur) return <p className="erreur">Configuration non lue : {config.erreur}</p>;
  const lenCible = (config.modeles_cible || []).map((m) => `${m.id}${m.max_model_len ? ` (contexte max ${m.max_model_len})` : ""}`).join(", ");
  const emp = config.empreinte;
  const dec = emp?.decodee;
  const base = (u) => u.replace(/\/metrics$/, "").replace(/\/$/, "");
  const nonVllm = !config.metriques_donnees && config.serveurs.every((s) => !s.version && !s.cache_config);
  return (
    <div className="config-serveur">
      <h4>Configuration du serveur</h4>
      {lenCible && <p>Modèles de la cible : {lenCible}</p>}
      {emp && (
        <p>Empreinte de la cible : <code>{emp.brute || "absente"}</code>
          {dec ? ` → TP ${dec.tensor_parallel_size}, PP ${dec.pipeline_parallel_size}, DP ${dec.data_parallel_size}, EP ${dec.enable_expert_parallel ? "oui" : "non"}`
               : emp.brute ? " (format non reconnu : --fingerprint-mode custom ?)" : emp.budget_epuise ? " (non relevée : budget de temps épuisé, serveurs lents)" : emp.reponse ? "" : " (pas de réponse : modèle, clé ou cible à vérifier)"}
          {dec?.masque_possible && " (ou masqué si vLLM tourne avec --fingerprint-mode hash)"}
        </p>
      )}
      {nonVllm && <p className="attention">La cible ne répond pas comme vLLM sur /version et /metrics : si c'est une passerelle (LiteLLM), donne les URL /metrics de vLLM.</p>}
      <table>
        <thead><tr><th>Serveur vLLM</th><th>Version</th><th>Cache de préfixes</th><th>TP</th><th>PP</th><th>DP</th><th>EP</th>
          <th>Décodage spéculatif</th><th>max_num_seqs</th><th>max_num_batched_tokens</th></tr></thead>
        <tbody>
          {config.serveurs.map((s) => {
            const cf = s.configuration || {};
            // Parallélisme : /server_info, sinon l'empreinte si ce serveur est la cible elle-même.
            const parEmp = !s.configuration && dec && base(s.metriques) === base(config.cible) ? dec : null;
            const par = s.configuration ? (cf.parallelisme || {}) : parEmp || {};
            const sp = cf.decodage_speculatif;
            const ord = cf.ordonnanceur || {};
            const pc = cf.cache?.enable_prefix_caching ?? s.cache_config?.enable_prefix_caching;
            const ndPar = s.configuration || parEmp ? "—" : "n/d";
            const nd = s.configuration ? "—" : "n/d";
            return (
              <tr key={s.metriques}>
                <td>{base(s.metriques)}</td>
                <td>{s.version || "?"}</td>
                <td>{oui(pc)}</td>
                <td>{par.tensor_parallel_size ?? ndPar}</td>
                <td>{par.pipeline_parallel_size ?? ndPar}</td>
                <td>{par.data_parallel_size ?? ndPar}</td>
                <td>{s.configuration || parEmp ? oui(par.enable_expert_parallel) : "n/d"}</td>
                <td>{s.configuration ? (sp ? `${sp.method ?? "?"}, ${sp.num_speculative_tokens ?? "?"} jetons` : "aucun")
                  : s.jetons_speculatifs_metriques != null ? (s.jetons_speculatifs_metriques > 0 ? `${s.jetons_speculatifs_metriques} jetons` : "aucun") : "n/d"}</td>
                <td>{ord.max_num_seqs ?? nd}</td>
                <td>{ord.max_num_batched_tokens ?? nd}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="aide">Parallélisme : lu dans l'empreinte system_fingerprint d'une complétion d'un jeton (mode par défaut de vLLM) ;
        via une passerelle, si elle la transmet, c'est celle de la réplique qui a répondu. Jetons spéculatifs : comptés dans /metrics.
        Méthode de décodage spéculatif et ordonnanceur (n/d) : seulement par /server_info, si vLLM est lancé avec
        VLLM_SERVER_DEV_MODE=1 — mode déconseillé : il ouvre des points d'accès de contrôle sans clé. Ce relevé est refait et
        enregistré au début de chaque essai (config_serveur.json).</p>
    </div>
  );
}

// ------------------------------------------------------------------ onglet Lancer
function Lancer({ surLance }) {
  const [type, setType] = useState("synthetique");
  const [c, setC] = useState(CONNEXION);
  const [params, setParams] = useState(INITIAL);
  const [msg, setMsg] = useState(null);
  const [test, setTest] = useState(null);
  const p = params[type];
  const majC = (k) => (e) => setC({ ...c, [k]: e.target.value });
  const maj = (k) => (e) => setParams({ ...params, [type]: { ...p, [k]: e.target.value } });
  const basculeAgent = (a) => setParams({ ...params, agents: { ...p,
    agents: p.agents.includes(a) ? p.agents.filter((x) => x !== a) : [...p.agents, a] } });

  const tester = async () => {
    setTest({ enCours: true });
    try {
      setTest(await api("/api/test-connexion", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(c) }));
    } catch (e) { setTest({ erreur: e.message }); }
  };
  const envoyer = async () => {
    setMsg(null);
    try {
      const r = await api("/api/essais", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, ...c, ...p }) });
      setMsg({ ok: true, texte: `Essai lancé : ${r.id}` });
      surLance(r.id);
    } catch (e) { setMsg({ ok: false, texte: e.message }); }
  };

  return (
    <section>
      <h3>Connexion</h3>
      <div className="grille">
        <Champ label="URL de la cible" aide="vLLM directement, ou une passerelle (LiteLLM) : http://hôte:port">
          <input value={c.url} onChange={majC("url")} />
        </Champ>
        <Champ label="Modèle" aide="nom servi par vLLM (--served-model-name) ou alias de la passerelle">
          <input value={c.modele} onChange={majC("modele")} />
        </Champ>
        <Champ label="URL des métriques vLLM" aide="optionnel ; à remplir si la cible est une passerelle, ou une URL par réplique (séparées par des espaces)">
          <input value={c.metriques} placeholder="http://hôte-vllm:8000/metrics" onChange={majC("metriques")} />
        </Champ>
        <Champ label="Clé d'API" aide="optionnel ; jamais écrite dans les journaux">
          <input type="password" autoComplete="off" value={c.cle_api} onChange={majC("cle_api")} />
        </Champ>
      </div>
      <div className="ligne">
        <button onClick={tester}>Tester la connexion</button>
        <div className="resultat-test">
          {test?.enCours && <span>…</span>}
          {test?.erreur && <span className="erreur">{test.erreur}</span>}
          {test?.cible && (
            <span className={test.cible.ok ? "ok" : "erreur"}>
              Cible : {test.cible.ok ? `répond, modèles ${test.cible.modeles.join(", ") || "—"}` : test.cible.erreur || `HTTP ${test.cible.statut}`}
            </span>
          )}
          {test?.metriques?.map((m) => (
            <span key={m.url} className={m.ok ? "ok" : "attention"}>
              Métriques {m.url} : {m.ok ? `${m.lignes_vllm} lignes vllm:` : m.erreur || `HTTP ${m.statut}, ${m.lignes_vllm ?? 0} ligne vllm:`}
            </span>
          ))}
        </div>
      </div>
      {test?.essai_en_cours && <p className="attention">Un essai est en cours : pas de complétion d'un jeton, donc pas d'empreinte (elle tomberait dans la mesure).</p>}
      {test?.config && <ConfigServeur config={test.config} />}

      <h3>Essai</h3>
      <div className="ligne">
        <div className="segments">
          <button className={type === "synthetique" ? "actif" : ""} onClick={() => { setType("synthetique"); setMsg(null); }}>Charge synthétique (AIPerf)</button>
          <button className={type === "agents" ? "actif" : ""} onClick={() => { setType("agents"); setMsg(null); }}>Agents de test</button>
        </div>
      </div>
      <div className="grille">
        <Champ label="Étiquette" aide="le suffixe -r1, -r2… marque une répétition">
          <input value={p.etiquette} onChange={maj("etiquette")} />
        </Champ>
        <Champ label={type === "synthetique" ? "Concurrences" : "Tâches simultanées"} aide="entiers séparés par des espaces">
          <input value={p.paliers} onChange={maj("paliers")} />
        </Champ>
        {type === "synthetique" ? (
          <>
            <Champ label="Tokenizer" aide="vide = outils/tokenizer du dépôt s'il existe, sinon le nom du modèle">
              <input value={p.tokenizer} placeholder="outils/tokenizer" onChange={maj("tokenizer")} />
            </Champ>
            <Champ label="Jetons d'entrée"><input type="number" value={p.isl} onChange={maj("isl")} /></Champ>
            <Champ label="Jetons de sortie"><input type="number" value={p.osl} onChange={maj("osl")} /></Champ>
            <Champ label="Durée par palier (s)" aide="vide = 8 × concurrence requêtes">
              <input type="number" value={p.duree} onChange={maj("duree")} />
            </Champ>
            <Champ label="Température"><input type="number" step="0.1" value={p.temperature} onChange={maj("temperature")} /></Champ>
            <Champ label="Dossier de résultats"><input value={p.racine} onChange={maj("racine")} /></Champ>
            <Champ label="Options AIPerf en plus" aide="ex. --use-server-token-count"><input value={p.extra} onChange={maj("extra")} /></Champ>
          </>
        ) : (
          <>
            <Champ label="Agents">
              <div className="cases">
                {[["revue_code", "Revue de code"], ["pentest", "Pentest pas à pas"], ["rapport", "Rédaction de rapport"]].map(([k, l]) => (
                  <label key={k}><input type="checkbox" checked={p.agents.includes(k)} onChange={() => basculeAgent(k)} /> {l}</label>
                ))}
              </div>
            </Champ>
            <Champ label="Mode" aide="scripté = mesure ; autonome = vérifie l'appel d'outils">
              <select value={p.mode} onChange={maj("mode")}>
                <option value="scripte">scripté</option><option value="autonome">autonome</option>
              </select>
            </Champ>
            <Champ label="Durée par palier (s)"><input type="number" value={p.duree} onChange={maj("duree")} /></Champ>
            <Champ label="Préchauffe exclue (s)"><input type="number" value={p.prechauffe} onChange={maj("prechauffe")} /></Champ>
            <Champ label="Échelle des tailles" aide="1 = tailles réelles ; plus petit pour un petit modèle"><input type="number" step="0.05" value={p.echelle} onChange={maj("echelle")} /></Champ>
            <Champ label="Facteur de latence des outils" aide="1 pour mesurer"><input type="number" step="0.1" value={p.facteur_latence} onChange={maj("facteur_latence")} /></Champ>
            <Champ label="Dossier de résultats"><input value={p.racine} onChange={maj("racine")} /></Champ>
          </>
        )}
      </div>
      <div className="ligne">
        <button className="principal" onClick={envoyer}>Lancer l'essai</button>
        {msg && <span className={msg.ok ? "ok" : "erreur"}>{msg.texte}</span>}
      </div>
      <p className="note">
        Le serveur vLLM se démarre à la main, avec les options du § 1.2 du README. Pour la charge synthétique, le cache de préfixes
        doit être désactivé ; pour les agents, activé.
      </p>
    </section>
  );
}

// ------------------------------------------------------------------ onglet Essais
function Essais({ selection, setSelection }) {
  const [essais, setEssais] = useState([]);
  const [texte, setTexte] = useState("");
  const pos = useRef(0);
  const boite = useRef(null);

  useEffect(() => {
    const charger = () => api("/api/essais").then(setEssais).catch(() => {});
    charger(); const t = setInterval(charger, 3000); return () => clearInterval(t);
  }, []);

  useEffect(() => {
    setTexte(""); pos.current = 0;
    if (!selection) return;
    let fini = false;
    const tirer = async () => {
      try {
        const r = await api(`/api/essais/${encodeURIComponent(selection)}/journal?depuis=${pos.current}`);
        if (r.texte) { pos.current = r.position; setTexte((t) => (t + r.texte).slice(-200000)); }
        if (r.termine && !r.texte) fini = true;
      } catch { fini = true; }
    };
    tirer(); const t = setInterval(() => { if (!fini) tirer(); }, 1500);
    return () => clearInterval(t);
  }, [selection]);

  useEffect(() => { if (boite.current) boite.current.scrollTop = boite.current.scrollHeight; }, [texte]);

  const arreter = (id) => api(`/api/essais/${encodeURIComponent(id)}/arreter`, { method: "POST" }).catch((e) => alert(e.message));
  const etat = (e) => (e.code === null ? "en cours" : e.code === 0 ? "terminé" : `échec (${e.code})`);

  return (
    <section className="deux-colonnes">
      <div>
        {essais.length === 0 && <p className="note">Aucun essai lancé depuis le démarrage de l'interface.</p>}
        <ul className="liste">
          {essais.map((e) => (
            <li key={e.id} className={selection === e.id ? "choisi" : ""} onClick={() => setSelection(e.id)}>
              <div><b>{e.etiquette}</b> <span className="pastille">{e.type === "agents" ? "agents" : "synthétique"}</span></div>
              <div className="petit">{new Date(e.debut * 1000).toLocaleTimeString()} · {etat(e)}
                {e.code === null && <button className="lien" onClick={(ev) => { ev.stopPropagation(); arreter(e.id); }}>arrêter</button>}
              </div>
            </li>
          ))}
        </ul>
      </div>
      <pre className="journal" ref={boite}>{selection ? texte || "…" : "Choisis un essai pour voir son journal."}</pre>
    </section>
  );
}

// ------------------------------------------------------------------ onglet Résultats
function Resultats() {
  const [racines, setRacines] = useState({ synthetique: [], agents: [] });
  const [type, setType] = useState("synthetique");
  const [racine, setRacine] = useState("");
  const [cible, setCible] = useState(30);
  const [gpu, setGpu] = useState(1);
  const [exiger, setExiger] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);

  const chargerRacines = () => api("/api/racines").then((r) => {
    setRacines(r);
    const l = r[type] || [];
    if (!l.includes(racine)) setRacine(l[0] || "");
  }).catch(() => {});
  useEffect(() => { chargerRacines(); }, [type]);

  const depouiller = async () => {
    setErr(null); setRes(null);
    try {
      const q = new URLSearchParams({ type, racine, cible, gpu, exiger: exiger ? "1" : "0" });
      setRes(await api(`/api/resultats?${q}`));
    } catch (e) { setErr(e.message); }
  };

  const agents = type === "agents";
  const lignes = (res?.lignes || []).filter((l) => !agents || l.agent === "tous");
  const essais = [...new Set(lignes.map((l) => l.essai))];
  const series = essais.map((e) => ({
    nom: e,
    points: lignes.filter((l) => l.essai === e)
      .map((l) => ({ x: +l.interactivite_p10, y: +l.debit_sortie_par_gpu, n: l.concurrence || l.taches_simultanees }))
      .filter((pt) => !isNaN(pt.x) && !isNaN(pt.y)).sort((a, b) => a.x - b.x),
  }));

  return (
    <section>
      <div className="grille">
        <Champ label="Type">
          <select value={type} onChange={(e) => { setType(e.target.value); setRes(null); }}>
            <option value="synthetique">Charge synthétique</option><option value="agents">Agents de test</option>
          </select>
        </Champ>
        <Champ label="Dossier de résultats" aide={<button className="lien" onClick={chargerRacines}>actualiser la liste</button>}>
          <select value={racine} onChange={(e) => setRacine(e.target.value)}>
            {(racines[type] || []).map((r) => <option key={r}>{r}</option>)}
          </select>
        </Champ>
        <Champ label="Interactivité visée (jetons/s par fil)"><input type="number" value={cible} onChange={(e) => setCible(e.target.value)} /></Champ>
        <Champ label="GPU derrière la cible (toutes répliques)"><input type="number" value={gpu} onChange={(e) => setGpu(e.target.value)} /></Champ>
        <Champ label="Exiger les métriques du serveur" aide="à cocher pour la mesure finale">
          <input type="checkbox" checked={exiger} onChange={(e) => setExiger(e.target.checked)} />
        </Champ>
      </div>
      <div className="ligne"><button className="principal" onClick={depouiller} disabled={!racine}>Dépouiller</button>
        {err && <span className="erreur">{err}</span>}</div>

      {res && (
        <>
          <div className="messages">
            {res.messages.length ? res.messages.map((m, i) => (
              <div key={i} className={m.includes("ATTENTION") ? "attention" : m.toLowerCase().includes("aucun") ? "erreur" : "retenu"}>{m.trim()}</div>
            )) : <pre>{res.sortie}</pre>}
          </div>

          <h3>Débit par GPU selon l'interactivité (10e centile)</h3>
          <div className="graphe">
            <ResponsiveContainer width="100%" height={340}>
              <LineChart margin={{ top: 10, right: 30, bottom: 30, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" name="interactivité P10"
                  domain={[(m) => Math.floor(Math.min(m, +cible) * 0.9), (m) => Math.ceil(Math.max(m, +cible) * 1.05)]}
                  label={{ value: "interactivité P10 (jetons/s par fil)", position: "bottom", offset: 10 }} />
                <YAxis type="number" dataKey="y" name="débit/GPU" label={{ value: "jetons/s par GPU", angle: -90, position: "insideLeft" }} />
                <Tooltip formatter={(v) => f1(v)} labelFormatter={(v) => `interactivité ${f1(v)}`} />
                <Legend verticalAlign="top" />
                <ReferenceLine x={+cible} stroke="#111" strokeDasharray="4 4" label={{ value: `cible ${cible}`, position: "top" }} />
                {series.map((s, i) => (
                  <Line key={s.nom} data={s.points} name={s.nom} dataKey="y" stroke={COULEURS[i % COULEURS.length]} dot isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <h3>Tableau</h3>
          <div className="tableau">
            <table>
              <thead>
                {agents ? (
                  <tr><th>essai</th><th>tâches simult.</th><th>agent</th><th>tâches</th><th>err</th><th>débit/GPU</th><th>inter P10</th><th>TTFT P95 s</th><th>cache</th><th>durée min</th><th>contexte max</th><th>préempt</th><th>MTP</th><th>cache KV max</th></tr>
                ) : (
                  <tr><th>essai</th><th>conc.</th><th>débit/GPU</th><th>inter P10</th><th>inter P50</th><th>TTFT P95 ms</th><th>préempt</th><th>err</th><th>cache</th><th>MTP</th><th>cache KV max</th><th>sortie moy</th></tr>
                )}
              </thead>
              <tbody>
                {(res.lignes || []).map((l, i) => agents ? (
                  <tr key={i} className={l.agent === "tous" ? "fort" : ""}>
                    <td>{l.essai}</td><td>{l.taches_simultanees}</td><td>{l.agent}</td><td>{l.taches_finies}</td>
                    <td>{(+l.taches_en_erreur || 0) + (+l.erreurs_appels || 0)}</td><td>{f1(l.debit_sortie_par_gpu)}</td>
                    <td>{f1(l.interactivite_p10)}</td><td>{f1(l.ttft_p95_s, 2)}</td><td>{pct(l.taux_cache)}</td>
                    <td>{f1(l.duree_tache_p50_min)}</td><td>{l.contexte_max}</td><td>{f1(l.preemptions, 0)}</td><td>{pct(l.acceptation_mtp)}</td><td>{pct(l.cache_kv_max_fraction)}</td>
                  </tr>
                ) : (
                  <tr key={i}>
                    <td>{l.essai}</td><td>{l.concurrence}</td><td>{f1(l.debit_sortie_par_gpu)}</td><td>{f1(l.interactivite_p10)}</td>
                    <td>{f1(l.interactivite_p50)}</td><td>{f1(l.ttft_p95_ms, 0)}</td><td>{f1(l.preemptions, 0)}</td>
                    <td>{l.requetes_en_erreur}</td><td>{pct(l.cache_prefixes)}</td><td>{pct(l.acceptation_mtp)}</td><td>{pct(l.cache_kv_max_fraction)}</td><td>{f1(l.sortie_moyenne_jetons, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">Commande exécutée : <code>{res.commande}</code></p>
        </>
      )}
    </section>
  );
}

export default function App() {
  const [onglet, setOnglet] = useState("lancer");
  const [selection, setSelection] = useState(null);
  return (
    <div className="page">
      {!JETON && <p className="erreur">Jeton absent : ouvre l'adresse complète affichée au démarrage de l'interface (…/?jeton=…).</p>}
      <header>
        <h1>Banc GPU</h1>
        <nav>
          {[["lancer", "Lancer"], ["essais", "Essais"], ["resultats", "Résultats"]].map(([k, l]) => (
            <button key={k} className={onglet === k ? "actif" : ""} onClick={() => setOnglet(k)}>{l}</button>
          ))}
        </nav>
      </header>
      {onglet === "lancer" && <Lancer surLance={(id) => { setSelection(id); setOnglet("essais"); }} />}
      {onglet === "essais" && <Essais selection={selection} setSelection={setSelection} />}
      {onglet === "resultats" && <Resultats />}
    </div>
  );
}
