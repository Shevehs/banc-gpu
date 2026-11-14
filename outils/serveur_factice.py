"""Faux serveur compatible OpenAI, pour vérifier les scripts du banc sans GPU ni modèle.
Imite : /v1/models, /v1/chat/completions en streaming (texte ou appel d'outil), le champ usage
avec les jetons lus en cache (préfixe commun avec les requêtes précédentes, 4 caractères par
jeton), /metrics au format Prometheus (quelques métriques vLLM), /version et, avec --mode-dev, /server_info. Les chiffres ne mesurent rien.
Usage : python serveur_factice.py --port 8001"""
import argparse
import asyncio, json, random
from aiohttp import web
VUS=[]  # requêtes déjà vues (texte), pour simuler le cache de préfixes
SORTIE=0; ENTREE=0; REQUETES=0; HTTP={"2xx":0}  # comme prometheus_fastapi_instrumentator : une série n'apparaît qu'à sa première requête
# Histogrammes imitant vLLM (bornes par défaut de vLLM ; --bornes-itl les remplace, comme --custom-histogram-buckets)
B_TTFT=[0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.25,0.5,0.75,1.0,2.5,5.0,7.5,10.0,20.0,40.0,80.0,160.0,640.0,2560.0]
B_ITL=[0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1.0,2.5,5.0,7.5,10.0,20.0,40.0,80.0]
H={"ttft":[], "tpot":[]}  # valeurs observées
# Imitation des métriques Prometheus de LiteLLM (--litellm), ventilées par api_key_alias = clé Bearer reçue.
# Format d'après https://docs.litellm.ai/docs/proxy/prometheus, comparé à LiteLLM 1.109.1 lors de la validation.
LL={}  # alias -> {"entree", "sortie", "ttft": [], "latence": []}
B_LL=[0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0,30.0,60.0,120.0,300.0]
def histo(nom, bornes, vals):
    t=f"# TYPE {nom} histogram\n"
    for b in bornes:
        t+=f'{nom}_bucket{{engine="0",le="{b}",model_name="banc"}} {sum(1 for v in vals if v<=b)}.0\n'
    t+=f'{nom}_bucket{{engine="0",le="+Inf",model_name="banc"}} {len(vals)}.0\n'
    t+=f'{nom}_count{{engine="0",model_name="banc"}} {len(vals)}.0\n{nom}_sum{{engine="0",model_name="banc"}} {sum(vals)}\n'
    return t
def texte_prompt(b):
    return json.dumps(b.get("messages",[]),ensure_ascii=False)+json.dumps(b.get("tools") or [])
def cache(t):
    best=0
    for v in VUS[-200:]:
        n=0; L=min(len(v),len(t))
        while n<L and v[n]==t[n]: n+=1
        best=max(best,n)
    VUS.append(t); return (best//4)//16*16
async def models(r): return web.json_response({"object":"list","data":[{"id":"banc","object":"model","max_model_len":32768}]})
async def chat(r):
    if random.random() < TAUX_ERREUR:
        HTTP["5xx"] = HTTP.get("5xx", 0) + 1
        return web.json_response({"error": "erreur simulée"}, status=500)
    b=await r.json(); t=texte_prompt(b); n_in=len(t)//4; c=cache(t)
    n=int(b.get("max_tokens") or b.get("max_completion_tokens") or 16); 
    if not b.get("ignore_eos"): n=max(1,int(n*random.uniform(0.3,1.0)))
    n=max(n,int(b.get("min_tokens") or 0)) if b.get("ignore_eos") else n
    if not b.get("stream"):
        return web.json_response({"id":"x","object":"chat.completion","model":"banc","system_fingerprint":EMPREINTE(),
            "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"length"}],
            "usage":{"prompt_tokens":n_in,"completion_tokens":1,"total_tokens":n_in+1}})
    resp=web.StreamResponse(headers={"Content-Type":"text/event-stream"}); await resp.prepare(r)
    t0=asyncio.get_running_loop().time()
    await asyncio.sleep(0.02+ (n_in-c)/2_000_000)
    t1=None
    outil = b.get("tools") and random.random()<0.7
    if outil:
        nom=random.choice(b["tools"])["function"]["name"]
        ch={"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":nom,"arguments":"{\"cible\":\"x\"}"}}]},"finish_reason":None}],"id":"x","object":"chat.completion.chunk","created":0,"model":"banc"}
        await resp.write(f"data: {json.dumps(ch)}\n\n".encode()); n=min(n,20)
    else:
        for i in range(n):
            ch={"choices":[{"index":0,"delta":{"content":f" w{i%50}"},"finish_reason":None}],"id":"x","object":"chat.completion.chunk","created":0,"model":"banc"}
            await resp.write(f"data: {json.dumps(ch)}\n\n".encode())
            if t1 is None: t1=asyncio.get_running_loop().time()
            await asyncio.sleep(ARGS.delai_jeton)
    global SORTIE, ENTREE, REQUETES; SORTIE += n; ENTREE += n_in; REQUETES += 1
    HTTP["2xx"] += 1
    t2=asyncio.get_running_loop().time()
    if t1 is not None:
        H["ttft"].append(t1-t0); H["tpot"].append((t2-t1)/(n-1) if n>1 else 0.0)
    alias=(r.headers.get("Authorization","").removeprefix("Bearer ").strip() or "sans-cle")[:40]
    meta=str(((b.get("metadata") or {}).get("agent")) or "")[:40]
    d=LL.setdefault((alias,meta),{"entree":0,"sortie":0,"ttft":[],"latence":[]})
    d["entree"]+=n_in; d["sortie"]+=n; d["latence"].append(t2-t0)
    if t1 is not None: d["ttft"].append(t1-t0)
    end={"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls" if outil else "length"}],"id":"x","object":"chat.completion.chunk","created":0,"model":"banc"}
    await resp.write(f"data: {json.dumps(end)}\n\n".encode())
    if (b.get("stream_options") or {}).get("include_usage"):
        u={"choices":[],"id":"x","object":"chat.completion.chunk","created":0,"model":"banc","system_fingerprint":EMPREINTE(),"usage":{"prompt_tokens":n_in,"completion_tokens":n,"total_tokens":n_in+n,"prompt_tokens_details":{"cached_tokens":c}}}
        await resp.write(f"data: {json.dumps(u)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n"); return resp
def histo_ll(nom, cle, vals):
    lab=f'api_key_alias="{cle[0]}",metadata_agent="{cle[1]}",model="banc"'
    t="".join(f'{nom}_bucket{{{lab},le="{b}"}} {sum(1 for v in vals if v<=b)}.0\n' for b in B_LL)
    return t+f'{nom}_bucket{{{lab},le="+Inf"}} {len(vals)}.0\n{nom}_sum{{{lab}}} {sum(vals)}\n{nom}_count{{{lab}}} {len(vals)}.0\n'
async def metrics_litellm(r):
    t=""
    for a,d in LL.items():
        lab=f'api_key_alias="{a[0]}",metadata_agent="{a[1]}",model="banc"'
        t+=f'litellm_input_tokens_metric_total{{{lab}}} {d["entree"]}.0\nlitellm_output_tokens_metric_total{{{lab}}} {d["sortie"]}.0\n'
        t+=histo_ll("litellm_llm_api_time_to_first_token_metric",a,d["ttft"])+histo_ll("litellm_request_total_latency_metric",a,d["latence"])
    return web.Response(text=t, content_type="text/plain")
async def metrics(r):
    t=("# TYPE vllm:generation_tokens_total counter\n"
       f"vllm:generation_tokens_total{{model_name=\"banc\"}} {SORTIE}\n"
       "# TYPE vllm:num_preemptions_total counter\n"
       "vllm:num_preemptions_total{model_name=\"banc\"} 0\n"
       "# TYPE vllm:kv_cache_usage_perc gauge\n"
       "vllm:kv_cache_usage_perc{model_name=\"banc\"} 0.5\n"
       "# TYPE vllm:prefix_cache_queries_total counter\n"
       f"vllm:prefix_cache_queries_total{{model_name=\"banc\"}} {SORTIE*10}\n"
       "# TYPE vllm:prefix_cache_hits_total counter\n"
       f"vllm:prefix_cache_hits_total{{model_name=\"banc\"}} {SORTIE}\n"
       "# TYPE vllm:spec_decode_num_draft_tokens_total counter\n"
       f"vllm:spec_decode_num_draft_tokens_total{{model_name=\"banc\"}} {SORTIE*3}\n"
       "# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n"
       f"vllm:spec_decode_num_accepted_tokens_total{{model_name=\"banc\"}} {SORTIE*2}\n"
       f"# TYPE vllm:prompt_tokens counter\nvllm:prompt_tokens_total{{engine=\"0\",model_name=\"banc\"}} {ENTREE}\n"
       f"# TYPE vllm:request_success counter\nvllm:request_success_total{{engine=\"0\",finished_reason=\"length\",model_name=\"banc\"}} {REQUETES}\n"
       + "".join(f'http_requests_total{{handler="/v1/chat/completions",method="POST",status="{k}"}} {v}.0\n' for k,v in HTTP.items()) +
       "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting{engine=\"0\",model_name=\"banc\"} 0.0\n"
       + histo("vllm:time_to_first_token_seconds", B_TTFT, H["ttft"])
       + histo("vllm:inter_token_latency_seconds", BORNES_ITL, H["tpot"])
       + histo("vllm:request_time_per_output_token_seconds", BORNES_ITL, H["tpot"]) +
       "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter\n"
       + "".join(f"vllm:spec_decode_num_accepted_tokens_per_pos_total{{model_name=\"banc\",engine=\"0\",position=\"{k}\"}} {SORTIE}\n" for k in range(3)) +
       "# TYPE vllm:cache_config_info gauge\n"
       "vllm:cache_config_info{block_size=\"16\",cache_dtype=\"auto\",enable_prefix_caching=\"True\",gpu_memory_utilization=\"0.9\",num_gpu_blocks=\"1000\"} 1.0\n")
    return web.Response(text=t, content_type="text/plain")
EMPREINTE=lambda: "vllm-0.0.0-tp4-dp2-ep-0a1b2c3d" if ARGS.mode_dev else "vllm-0.0.0-tp8-ep-0a1b2c3d"
async def version(r): return web.json_response({"version":"0.0.0-factice"})
async def server_info(r):
    if not ARGS.mode_dev: return web.Response(status=404)
    return web.json_response({"vllm_config":{"parallel_config":{"tensor_parallel_size":4,"pipeline_parallel_size":1,"data_parallel_size":2,"enable_expert_parallel":True,"decode_context_parallel_size":1},
        "model_config":{"model":"/poids/factice","served_model_name":"banc","max_model_len":32768},"cache_config":{"enable_prefix_caching":True,"block_size":16},
        "scheduler_config":{"max_num_seqs":256,"max_num_batched_tokens":8192,"enable_chunked_prefill":True},"speculative_config":{"method":"mtp","num_speculative_tokens":3}}})
a=argparse.ArgumentParser()
a.add_argument("--delai-jeton",type=float,default=0.003,help="secondes entre deux jetons (règle la vitesse simulée)")
a.add_argument("--bornes-itl",default="",help="bornes JSON de l'histogramme inter_token_latency, comme --custom-histogram-buckets de vLLM")
a.add_argument("--mode-dev",action="store_true",help="imite VLLM_SERVER_DEV_MODE=1 (/server_info)"); a.add_argument("--port",type=int,default=8001); a.add_argument("--taux-erreur",type=float,default=0.0)
ARGS=a.parse_args(); BORNES_ITL=json.loads(ARGS.bornes_itl) if ARGS.bornes_itl else B_ITL; PORT=ARGS.port; TAUX_ERREUR=ARGS.taux_erreur
app=web.Application(client_max_size=200*1024*1024); app.add_routes([web.get("/v1/models",models),web.post("/v1/chat/completions",chat),web.get("/metrics",metrics),web.get("/version",version),web.get("/server_info",server_info),web.get("/litellm/metrics/",metrics_litellm),
    web.get("/litellm/metrics",lambda r: web.HTTPTemporaryRedirect("/litellm/metrics/"))])
web.run_app(app,host="127.0.0.1",port=PORT,print=None)
