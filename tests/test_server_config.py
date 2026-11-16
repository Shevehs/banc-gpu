"""tools/server_config.py: fingerprint decoding, configuration extraction, summary and collection against a stub."""
import json
import urllib.error

import pytest

from tests import helpers
from tools import server_config
from tools.server_config import collect, read_fingerprint, summary

METRICS = """# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",gpu_memory_utilization="0.9",num_gpu_blocks="1000",other="x"} 1.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",engine="0",position="0"} 5.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",engine="0",position="1"} 4.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",engine="0",position="2"} 3.0
"""
INFO = {"vllm_config": {"parallel_config": {"tensor_parallel_size": 4, "data_parallel_size": 2, "secret": "x"},
                        "model_config": {"model": "/w", "max_model_len": 32768, "unknown": 1},
                        "cache_config": {"block_size": 16}, "scheduler_config": {"max_num_seqs": 256},
                        "speculative_config": {"method": "mtp", "num_speculative_tokens": 3, "other": 0}}}


@pytest.mark.parametrize("raw,expected", [
    ("vllm-0.11.0-tp4-dp2-ep-0a1b2c3d", dict(version="0.11.0", tensor_parallel_size=4, pipeline_parallel_size=1,
                                              data_parallel_size=2, enable_expert_parallel=True, hash_config="0a1b2c3d",
                                              masking_possible=False)),
    ("vllm-0.11.0-0a1b2c3d", dict(version="0.11.0", tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                                  enable_expert_parallel=False, hash_config="0a1b2c3d", masking_possible=True)),
    ("vllm-0.11.0rc1.dev5+gabc-tp8-pp2-nohash", dict(version="0.11.0rc1.dev5+gabc", tensor_parallel_size=8,
                                                     pipeline_parallel_size=2, data_parallel_size=1,
                                                     enable_expert_parallel=False, hash_config="nohash",
                                                     masking_possible=False)),
])
def test_read_fingerprint(raw, expected):
    assert read_fingerprint(raw) == expected


@pytest.mark.parametrize("raw", [None, 5, "", "gpt-4", "vllm-", "vllm-0.1.2", "vllm-0.1.2-short", "x" * 301,
                                  "vllm-0.1.2-tp4-0a1b2c3d\nextra", ["vllm-0.1.2-0a1b2c3d"]])
def test_read_fingerprint_rejects(raw):
    assert read_fingerprint(raw) is None


def test_configuration_keeps_only_known_fields():
    cfg = server_config._configuration(INFO["vllm_config"])
    assert cfg["parallelism"] == {"tensor_parallel_size": 4, "data_parallel_size": 2}
    assert cfg["model"] == {"model": "/w", "max_model_len": 32768} and cfg["cache"] == {"block_size": 16}
    assert cfg["scheduler"] == {"max_num_seqs": 256}
    assert cfg["speculative_decoding"] == {"method": "mtp", "num_speculative_tokens": 3}


def test_configuration_tolerates_garbage():
    assert server_config._configuration(None) is None and server_config._configuration([]) is None
    cfg = server_config._configuration({"parallel_config": "x", "speculative_config": None})
    assert cfg["parallelism"] == {} and cfg["speculative_decoding"] is None


def routes(**over):
    base = {"/v1/models": (200, {"data": [{"id": "bench", "max_model_len": 32768}, "junk"]}, {}),
            "/version": (200, {"version": "0.11.0"}, {}), "/metrics": (200, METRICS, {}),
            "/server_info": (200, INFO, {}),
            "/v1/chat/completions": (200, {"system_fingerprint": "vllm-0.11.0-tp4-dp2-ep-0a1b2c3d"}, {})}
    base.update(over)
    return base


def test_collect_full(stub):
    s = stub(routes())
    r = collect(s.url, [s.url + "/metrics"], key="sk-1", model="bench")
    assert r["target"] == s.url and r["given_metrics"] is True
    assert r["target_models"] == [{"id": "bench", "max_model_len": 32768}]
    (srv,) = r["servers"]
    assert srv["version"] == "0.11.0" and srv["speculative_tokens_metrics"] == 3
    assert srv["vllm_lines"] == 4 and srv["metrics_status"] == 200      # vllm: lines read in /metrics
    assert srv["cache_config"] == {"block_size": "16", "cache_dtype": "auto", "enable_prefix_caching": "True",
                                   "gpu_memory_utilization": "0.9", "num_gpu_blocks": "1000"}
    assert srv["configuration"]["parallelism"]["tensor_parallel_size"] == 4
    fp = r["fingerprint"]
    assert fp["raw"] == "vllm-0.11.0-tp4-dp2-ep-0a1b2c3d" and fp["response"] is True
    assert fp["decoded"]["tensor_parallel_size"] == 4 and fp["decoded"]["enable_expert_parallel"] is True
    json.dumps(r)


def test_collect_sends_the_key_to_the_target_only(stub):
    s = stub(routes())
    collect(s.url, [s.url + "/metrics"], key="sk-1", model="bench")
    by_path = {r["path"].split("?")[0]: r["headers"].get("Authorization") for r in s.seen}
    assert by_path["/v1/models"] == "Bearer sk-1" and by_path["/v1/chat/completions"] == "Bearer sk-1"
    assert by_path["/metrics"] is None and by_path["/version"] is None and by_path["/server_info"] is None


def test_collect_chat_request_is_one_token(stub):
    s = stub(routes())
    collect(s.url, [], model="bench")
    body = json.loads([r for r in s.seen if r["method"] == "POST"][0]["body"])
    assert body["max_tokens"] == 1 and body["model"] == "bench" and body["temperature"] == 0


def test_collect_default_metrics_url_and_no_model(stub):
    s = stub(routes())
    r = collect(s.url + "/", None)
    assert r["given_metrics"] is False and r["servers"][0]["metrics"] == s.url + "/metrics" and r["fingerprint"] is None
    assert not any(x["method"] == "POST" for x in s.seen)


def test_collect_does_not_raise_on_an_unusable_server(stub):
    s = stub({"/v1/models": (500, "", {}), "/metrics": (200, "garbage", {}), "/version": (200, "not json", {})})
    r = collect(s.url, [s.url + "/metrics"], model="bench")
    assert r["target_models"] is None and r["servers"][0]["version"] is None
    assert r["servers"][0]["speculative_tokens_metrics"] is None and r["servers"][0]["configuration"] is None
    assert r["servers"][0]["vllm_lines"] == 0 and r["servers"][0]["metrics_status"] == 200
    assert r["fingerprint"] == {"raw": None, "decoded": None, "response": False}


def test_collect_unreachable_target():
    r = collect(f"http://127.0.0.1:{helpers.free_port()}", model="bench", budget_s=3)
    assert r["target_models"] is None and r["servers"][0]["version"] is None and r["fingerprint"]["response"] is False


def test_collect_does_not_follow_redirects(stub):
    s = stub(routes(**{"/v1/models": (302, "", {"Location": "/other"}), "/other": (200, {"data": []}, {})}))
    assert collect(s.url)["target_models"] is None
    assert all(r["path"] != "/other" for r in s.seen)


def test_collect_budget_exhausted(stub):
    s = stub(routes())
    r = collect(s.url, [s.url + "/metrics"], model="bench", budget_s=0)
    assert r["fingerprint"]["budget_exhausted"] is True and r["servers"][0]["version"] is None
    assert not any(x["path"] == "/version" for x in s.seen)


def test_collect_metrics_without_vllm_lines_gives_unknown_spec_tokens(stub):
    s = stub(routes(**{"/metrics": (200, "other_metric 1\n", {})}))
    srv = collect(s.url)["servers"][0]
    assert srv["speculative_tokens_metrics"] is None and srv["vllm_lines"] == 0 and srv["metrics_status"] == 200


def test_collect_counts_only_vllm_prefixed_lines(stub):
    text = "# vllm:comment\nvllm:a 1\n  vllm:b 2\nvllm:c{x=\"1\"} 3\nmy_vllm:d 4\n"
    assert collect(stub(routes(**{"/metrics": (200, text, {})})).url)["servers"][0]["vllm_lines"] == 2


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_collect_metrics_http_error_is_recorded_not_raised(stub, status):
    srv = collect(stub(routes(**{"/metrics": (status, "no", {})})).url)["servers"][0]
    assert srv["metrics_status"] == status and srv["vllm_lines"] is None and srv["speculative_tokens_metrics"] is None


def test_collect_metrics_redirect_is_recorded_as_its_status(stub):
    s = stub(routes(**{"/metrics": (307, "", {"Location": "/metrics/"}), "/metrics/": (200, METRICS, {})}))
    srv = collect(s.url)["servers"][0]
    assert srv["metrics_status"] == 307 and srv["vllm_lines"] is None
    assert all(r["path"] != "/metrics/" for r in s.seen)


def test_collect_unreachable_metrics_has_no_status(stub):
    srv = collect(stub(routes()).url, [f"http://127.0.0.1:{helpers.free_port()}/metrics"])["servers"][0]
    assert srv["metrics_status"] is None and srv["vllm_lines"] is None


def test_collect_budget_exhausted_marks_the_metrics_not_read(stub):
    srv = collect(stub(routes()).url, [], budget_s=0)["servers"][0]
    assert srv["metrics_status"] == "budget" and srv["vllm_lines"] is None


def test_collect_one_entry_per_metrics_url(stub):
    a, b = stub(routes()), stub(routes(**{"/metrics": (200, "x 1\n", {})}))
    r = collect(a.url, [a.url + "/metrics", b.url + "/metrics"])
    assert [x["metrics"] for x in r["servers"]] == [a.url + "/metrics", b.url + "/metrics"]
    assert [x["vllm_lines"] for x in r["servers"]] == [4, 0]


def test_summary(stub):
    s = stub(routes())
    text = summary(collect(s.url, [s.url + "/metrics"], model="bench"))
    assert "target fingerprint: vllm-0.11.0-tp4-dp2-ep-0a1b2c3d -> TP 4, PP 1, DP 2, EP yes" in text
    assert "vLLM 0.11.0; prefix cache True; speculative tokens (metrics) 3; /server_info read (speculative decoding:" in text
    assert "[config] GPUs deduced: 8 (TP 4 × PP 1 × DP 2 × 1 vLLM server(s))" in text
    assert text.count("\n") == 2 and "WARNING" not in text


def test_summary_variants():
    base = {"servers": [{"metrics": "m", "version": None, "cache_config": None, "configuration": None,
                         "speculative_tokens_metrics": None, "vllm_lines": 5, "metrics_status": 200}], "fingerprint": None}
    assert summary(base) == "[config] m: vLLM ?; prefix cache ?; speculative tokens (metrics) ?; /server_info " \
        "absent or not read (VLLM_SERVER_DEV_MODE not enabled?)"
    assert "time budget exhausted" in summary({**base, "fingerprint": {"budget_exhausted": True}})
    no_answer = summary({**base, "fingerprint": {"raw": None, "decoded": None, "response": False}})
    assert "absent" in no_answer and "(no response: check the model, key or target)" in no_answer
    masked = {"raw": "vllm-1-0a1b2c3d", "response": True, "decoded": read_fingerprint("vllm-1-0a1b2c3d")}
    assert "(or masked if --fingerprint-mode hash)" in summary({**base, "fingerprint": masked})


def server(**kw):
    return {"metrics": "http://h:8000/metrics", "version": "0.11.0", "cache_config": None, "configuration": None,
            "speculative_tokens_metrics": None, "vllm_lines": 10, "metrics_status": 200, **kw}


def test_summary_gpus_deduced_line_names_the_origin_and_the_server_count():
    fp = {"raw": "vllm-1-tp2-pp2-0a1b2c3d", "response": True, "decoded": read_fingerprint("vllm-1-tp2-pp2-0a1b2c3d")}
    text = summary({"servers": [server(), server(metrics="http://h2:8000/metrics")], "fingerprint": fp})
    assert "[config] GPUs deduced: 8 (TP 2 × PP 2 × DP 1 × 2 vLLM server(s))" in text


def test_summary_gpus_line_warns_that_a_masked_fingerprint_assumes_one_gpu_per_server():
    fp = {"raw": "vllm-1-0a1b2c3d", "response": True, "decoded": read_fingerprint("vllm-1-0a1b2c3d")}
    text = summary({"servers": [server()], "fingerprint": fp})
    assert "[config] GPUs deduced: 1 (TP 1 × PP 1 × DP 1 × 1 vLLM server(s)); 1 GPU per server unless" in text


def test_summary_has_no_gpus_line_when_nothing_can_be_deduced():
    fp = {"raw": None, "response": False, "decoded": None}
    assert "GPUs deduced" not in summary({"servers": [server()], "fingerprint": fp})
    assert "GPUs deduced" not in summary({"servers": [server()], "fingerprint": None})
    fp = {"raw": "vllm-1-tp2-0a1b2c3d", "response": True, "decoded": read_fingerprint("vllm-1-tp2-0a1b2c3d")}
    assert "GPUs deduced" not in summary({"servers": [server(vllm_lines=0)], "fingerprint": fp})


@pytest.mark.parametrize("changes,fragment", [
    ({"vllm_lines": 0}, "no vllm: metric, LiteLLM address?"),
    ({"vllm_lines": None, "metrics_status": 401}, "HTTP 401: key required"),
    ({"vllm_lines": None, "metrics_status": 403}, "HTTP 403: key required"),
    ({"vllm_lines": None, "metrics_status": 307}, "HTTP 307: key required"),
    ({"vllm_lines": None, "metrics_status": 308}, "HTTP 308: key required"),
])
def test_summary_warns_when_metrics_are_not_vllms(changes, fragment):
    text = summary({"servers": [server(**changes)], "fingerprint": None})
    assert "[config] WARNING: http://h:8000/metrics is not vLLM's /metrics (" + fragment in text
    assert "put vLLM's /metrics address in BENCH_METRICS" in text


def test_summary_warns_when_the_time_budget_ran_out_or_the_server_is_unreachable():
    text = summary({"servers": [server(vllm_lines=None, metrics_status="budget")], "fingerprint": None})
    assert "WARNING: http://h:8000/metrics not read (time budget exhausted, slow servers)." in text
    text = summary({"servers": [server(vllm_lines=None, metrics_status=500)], "fingerprint": None})
    assert "WARNING: http://h:8000/metrics unreachable (HTTP 500): preemptions, cache and KV cache will not be read." in text
    text = summary({"servers": [server(vllm_lines=None, metrics_status=None)], "fingerprint": None})
    assert "WARNING: http://h:8000/metrics unreachable: preemptions" in text


def test_summary_no_warning_for_a_healthy_server_and_for_files_without_the_new_keys():
    assert "WARNING" not in summary({"servers": [server()], "fingerprint": None})
    old = {"metrics": "m", "version": "1", "cache_config": None, "configuration": None, "speculative_tokens_metrics": None}
    assert "WARNING: m unreachable" in summary({"servers": [old], "fingerprint": None})     # no vllm_lines key: unknown


def test_summary_warning_per_server():
    text = summary({"servers": [server(), server(metrics="http://lite:4000/metrics", vllm_lines=0)], "fingerprint": None})
    assert text.count("WARNING") == 1 and "http://lite:4000/metrics is not vLLM's" in text


def test_main_writes_the_snapshot_and_sends_the_bench_key(stub, tmp_path):
    s = stub(routes())
    out = tmp_path / "cfg.json"
    r = helpers.run([helpers.sys.executable, "-B", "tools/server_config.py", "--url", s.url, "--model", "bench",
                     "--metrics", s.url + "/metrics", "--output", str(out)],
                    env=helpers.clean_environment(BENCH_API_KEY="sk-bench", API_KEY="sk-generic"))
    assert r.returncode == 0 and "[config]" in r.stderr and r.stdout == ""
    assert json.loads(out.read_text())["fingerprint"]["response"] is True
    keys = {x["headers"].get("Authorization") for x in s.seen if x["path"] == "/v1/models"}
    assert keys == {"Bearer sk-bench"}


def test_main_prints_json_without_output(stub):
    s = stub(routes())
    r = helpers.run([helpers.sys.executable, "-B", "tools/server_config.py", "--url", s.url])
    assert json.loads(r.stdout)["target"] == s.url


# ---------------------------------------------------------------- --check-model
UNKNOWN_VLLM = (404, {"error": {"message": "The model `bench` does not exist. Valid aliases: a.",
                                "type": "NotFoundError", "param": "model", "code": 404}}, {})
UNKNOWN_LITELLM = (400, {"error": {"message": "/chat/completions: Invalid model name passed in model=bench.",
                                   "type": "invalid_request_error", "param": None, "code": "400"}}, {})
UNKNOWN_OPENAI = (404, {"error": {"message": "The model `bench` does not exist or you do not have access to it.",
                                  "type": "invalid_request_error", "param": None, "code": "model_not_found"}}, {})


@pytest.mark.parametrize("answer", [UNKNOWN_VLLM, UNKNOWN_LITELLM, UNKNOWN_OPENAI])
def test_unknown_model_message_is_the_servers_own(stub, answer):
    s = stub(routes(**{"/v1/chat/completions": answer}))
    assert server_config.unknown_model_message(s.url, "bench") == answer[1]["error"]["message"]


@pytest.mark.parametrize("answer", [
    (200, {"system_fingerprint": "x"}, {}),                                          # accepted
    (200, "not json", {}),                                                           # accepted, whatever the body
    (401, {"error": {"message": "Authentication Error, No api key passed in."}}, {}),  # the key is refused, not the model
    (403, {"error": {"message": "model not allowed for this key"}}, {}),             # 403 is not "unknown model"
    (404, "Not Found", {}),                                                          # wrong address
    (404, {"detail": "Not Found"}, {}),
    (400, {"error": {"message": "max_tokens must be positive"}}, {}),                # another bad request
    (400, {"error": {"message": "This model's maximum context length is 10 tokens."}}, {}),   # names the model, is not unknown
    (500, {"error": {"message": "model does not exist (internal)"}}, {}),            # a server error
    (302, "", {"Location": "/elsewhere"}),                                           # redirects are refused, not followed
])
def test_unknown_model_message_ignores_every_other_answer(stub, answer):
    s = stub(routes(**{"/v1/chat/completions": answer}))
    assert server_config.unknown_model_message(s.url, "bench") is None


def test_unknown_model_message_when_nothing_answers():
    assert server_config.unknown_model_message("http://127.0.0.1:1", "bench", timeout_s=1) is None


def test_unknown_model_message_sends_the_key_and_one_token(stub):
    s = stub(routes(**{"/v1/chat/completions": UNKNOWN_VLLM}))
    server_config.unknown_model_message(s.url + "/", "bench", key="sk-1")
    (request,) = [r for r in s.seen if r["method"] == "POST"]
    assert request["headers"]["Authorization"] == "Bearer sk-1" and request["path"] == "/v1/chat/completions"
    body = json.loads(request["body"])
    assert body["model"] == "bench" and body["max_tokens"] == 1 and body["temperature"] == 0


def test_check_model_passes_silently_for_a_known_model(stub, capsys):
    s = stub(routes())
    assert server_config.check_model(s.url, "bench") == 0
    assert capsys.readouterr().err == ""
    assert [r["path"] for r in s.seen] == ["/v1/chat/completions"]            # one request, nothing else


def test_check_model_reports_the_server_message_and_the_served_models(stub, capsys):
    s = stub(routes(**{"/v1/chat/completions": UNKNOWN_VLLM,
                       "/v1/models": (200, {"data": [{"id": "a"}, {"id": "b"}, "junk", {"id": 3}]}, {})}))
    assert server_config.check_model(s.url, "bench", key="sk-1") == 3
    err = capsys.readouterr().err
    assert 'ERROR: the server does not know the model "bench": The model `bench` does not exist.' in err
    assert "Models served: a, b" in err and "…" not in err
    assert {r["headers"].get("Authorization") for r in s.seen} == {"Bearer sk-1"}


def test_check_model_shortens_a_long_list_of_served_models(stub, capsys):
    models = {"data": [{"id": f"m{i}"} for i in range(12)]}
    s = stub(routes(**{"/v1/chat/completions": UNKNOWN_VLLM, "/v1/models": (200, models, {})}))
    assert server_config.check_model(s.url, "bench") == 3
    assert "m9 …" in capsys.readouterr().err


def test_check_model_without_a_list_still_reports_the_message(stub, capsys):
    s = stub(routes(**{"/v1/chat/completions": UNKNOWN_LITELLM, "/v1/models": (401, {}, {})}))
    assert server_config.check_model(s.url, "bench") == 3
    err = capsys.readouterr().err
    assert "Invalid model name" in err and "Models served" not in err


def test_main_check_model_exit_codes(stub):
    bad = stub(routes(**{"/v1/chat/completions": UNKNOWN_VLLM}))
    good = stub(routes())
    command = [helpers.sys.executable, "-B", "tools/server_config.py", "--check-model", "--model", "bench", "--url"]
    refused = helpers.run(command + [bad.url])
    assert refused.returncode == 3 and "does not know the model" in refused.stderr and refused.stdout == ""
    assert helpers.run(command + [good.url]).returncode == 0
    assert helpers.run(command + ["http://127.0.0.1:1"]).returncode == 0           # cannot ask: no verdict


def test_main_check_model_needs_a_model(stub):
    s = stub(routes())
    r = helpers.run([helpers.sys.executable, "-B", "tools/server_config.py", "--check-model", "--url", s.url])
    assert r.returncode == 2 and "--check-model needs --model" in r.stderr


@pytest.mark.parametrize("answer,expected", [
    ((400, {"error": {"message": "Tokenizer for model x not found"}}, {}), None),      # a 400 that names a model for another reason
    ((400, {"error": {"message": "Chat template for model x does not exist"}}, {}), None),
    ((400, {"error": {"message": "unknown", "code": "model_not_found"}}, {}), "unknown"),   # the code is enough, with a 400 or a 404
    ((404, {"error": {"message": "gone", "code": "model_not_found"}}, {}), "gone"),
    ((500, {"error": {"message": "gone", "code": "model_not_found"}}, {}), None),
    ((404, {"error": "model bench not found"}, {}), "model bench not found"),          # error given as a plain string
    ((404, {"error": ["model bench not found"]}, {}), '{"error": ["model bench not found"]}'),   # another shape: the text
    ((404, {"error": ["something else"]}, {}), None),
    ((404, "model bench not found", {}), "model bench not found"),                     # not JSON at all, but it says so
])
def test_unknown_model_message_edge_cases(stub, answer, expected):
    s = stub(routes(**{"/v1/chat/completions": answer}))
    assert server_config.unknown_model_message(s.url, "bench") == expected


def test_unknown_model_message_fits_on_one_line(stub):
    message = "The model `bench`\n\tdoes not exist.   " + "x" * 1000
    s = stub(routes(**{"/v1/chat/completions": (404, {"error": {"message": message, "param": "model"}}, {})}))
    result = server_config.unknown_model_message(s.url, "bench")
    assert result.startswith("The model `bench` does not exist. xxx") and len(result) == 300
    assert "\n" not in result and "\t" not in result


def test_unknown_model_message_survives_an_error_body_that_cannot_be_read(monkeypatch):
    class Broken:
        def read(self, *a):
            raise OSError("connection reset")

        def close(self):
            pass

    def refuse(*a, **k):
        raise urllib.error.HTTPError("http://x", 404, "Not Found", {}, Broken())
    monkeypatch.setattr(server_config, "_request", refuse)
    assert server_config.unknown_model_message("http://x", "bench") is None


def test_collect_records_a_refused_model_and_the_summary_says_so(stub):
    s = stub(routes(**{"/v1/chat/completions": UNKNOWN_VLLM}))
    r = collect(s.url, [s.url + "/metrics"], model="bench")
    assert r["fingerprint"]["response"] is False
    assert r["fingerprint"]["unknown_model"] == UNKNOWN_VLLM[1]["error"]["message"]
    assert "[config] ERROR: the server does not know the model: The model `bench` does not exist." in summary(r)
    assert len([x for x in s.seen if x["method"] == "POST"]) == 1                 # the same single request


def test_collect_of_an_accepted_model_has_no_refusal(stub):
    s = stub(routes())
    r = collect(s.url, [s.url + "/metrics"], model="bench")
    assert "unknown_model" not in r["fingerprint"] and "ERROR" not in summary(r)
