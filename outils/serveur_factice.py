"""Faux serveur compatible OpenAI, pour vérifier les scripts du banc sans GPU ni modèle.
Imite : /v1/models, /v1/chat/completions en streaming (texte ou appel d'outil), le champ usage
avec les jetons lus en cache (préfixe commun avec les requêtes précédentes, 4 caractères par
jeton), /metrics au format Prometheus (quelques métriques vLLM), /version et, avec --mode-dev, /server_info. Les chiffres ne mesurent rien.
Usage : python serveur_factice.py --port 8001"""
import argparse
import asyncio, json, random
from aiohttp import web
VUS=[]  # requêtes déjà vues (texte), pour simuler le cache de préfixes
SORTIE=0
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
        return web.json_response({"error": "erreur simulée"}, status=500)
    b=await r.json(); t=texte_prompt(b); n_in=len(t)//4; c=cache(t)
    n=int(b.get("max_tokens") or 16); 
    if not b.get("ignore_eos"): n=max(1,int(n*random.uniform(0.3,1.0)))
    n=max(n,int(b.get("min_tokens") or 0)) if b.get("ignore_eos") else n
    if not b.get("stream"):
        return web.json_response({"id":"x","object":"chat.completion","model":"banc","system_fingerprint":EMPREINTE(),
            "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"length"}],
            "usage":{"prompt_tokens":n_in,"completion_tokens":1,"total_tokens":n_in+1}})
    resp=web.StreamResponse(headers={"Content-Type":"text/event-stream"}); await resp.prepare(r)
    await asyncio.sleep(0.02+ (n_in-c)/2_000_000)
    outil = b.get("tools") and random.random()<0.7
    if outil:
        nom=random.choice(b["tools"])["function"]["name"]
        ch={"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":nom,"arguments":"{\"cible\":\"x\"}"}}]},"finish_reason":None}],"id":"x","object":"chat.completion.chunk","created":0,"model":"banc"}
        await resp.write(f"data: {json.dumps(ch)}\n\n".encode()); n=min(n,20)
    else:
        for i in range(n):
            ch={"choices":[{"index":0,"delta":{"content":f" w{i%50}"},"finish_reason":None}],"id":"x","object":"chat.completion.chunk","created":0,"model":"banc"}
            await resp.write(f"data: {json.dumps(ch)}\n\n".encode()); await asyncio.sleep(0.003)
    global SORTIE; SORTIE += n
    end={"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls" if outil else "length"}],"id":"x","object":"chat.completion.chunk","created":0,"model":"banc"}
    await resp.write(f"data: {json.dumps(end)}\n\n".encode())
    if (b.get("stream_options") or {}).get("include_usage"):
        u={"choices":[],"id":"x","object":"chat.completion.chunk","created":0,"model":"banc","system_fingerprint":EMPREINTE(),"usage":{"prompt_tokens":n_in,"completion_tokens":n,"total_tokens":n_in+n,"prompt_tokens_details":{"cached_tokens":c}}}
        await resp.write(f"data: {json.dumps(u)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n"); return resp
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
a=argparse.ArgumentParser(); a.add_argument("--mode-dev",action="store_true",help="imite VLLM_SERVER_DEV_MODE=1 (/server_info)"); a.add_argument("--port",type=int,default=8001); a.add_argument("--taux-erreur",type=float,default=0.0)
ARGS=a.parse_args(); PORT=ARGS.port; TAUX_ERREUR=ARGS.taux_erreur
app=web.Application(client_max_size=200*1024*1024); app.add_routes([web.get("/v1/models",models),web.post("/v1/chat/completions",chat),web.get("/metrics",metrics),web.get("/version",version),web.get("/server_info",server_info)])
web.run_app(app,host="127.0.0.1",port=PORT,print=None)
