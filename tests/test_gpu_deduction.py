"""GPU deduction from the fingerprint: tools.server_config.deduced_gpus/run_gpus, report.common.GpuResolver and the
reports that use it (--gpu optional, contradiction and repetition warnings, error when unknown)."""
import json

import pytest

from report import common
from report.cli import main
from report.common import GpuResolver
from tests import fixtures
from tools.server_config import deduced_gpus, read_fingerprint, run_gpus

FINGERPRINT = "vllm-0.11.0-tp4-dp2-ep-0a1b2c3d"


def config(fingerprint=FINGERPRINT, servers=1, pcp=None, **server):
    """The server_config.json of a run: a decoded fingerprint and `servers` vLLM servers (10 vllm: lines each)."""
    entry = {"metrics": "http://h:8000/metrics", "version": "0.11.0", "cache_config": None, "configuration": None,
             "speculative_tokens_metrics": None, "vllm_lines": 10, "metrics_status": 200, **server}
    if pcp is not None:
        entry["configuration"] = {"parallelism": {"prefill_context_parallel_size": pcp}}
    return {"target": "http://t", "servers": [dict(entry, metrics=f"http://h{i}:8000/metrics") for i in range(servers)],
            "fingerprint": {"raw": fingerprint, "response": True, "decoded": read_fingerprint(fingerprint)}}


# ---------------------------------------------------------------- deduced_gpus
@pytest.mark.parametrize("fingerprint,servers,expected", [
    ("vllm-0.11.0-tp4-dp2-ep-0a1b2c3d", 1, 8),
    ("vllm-0.11.0-tp8-0a1b2c3d", 1, 8),
    ("vllm-0.11.0-tp2-pp2-0a1b2c3d", 1, 4),
    ("vllm-0.11.0-tp2-pp2-dp3-0a1b2c3d", 1, 12),
    ("vllm-0.11.0-dp4-0a1b2c3d", 1, 4),
    ("vllm-0.11.0-pp2-0a1b2c3d", 1, 2),
    ("vllm-0.11.0-tp4-0a1b2c3d", 3, 12),
    ("vllm-0.11.0-ep-0a1b2c3d", 1, 1),
    ("vllm-0.11.0-0a1b2c3d", 1, 1),
    ("vllm-0.11.0-0a1b2c3d", 2, 2),
])
def test_count_is_tp_times_pp_times_dp_times_the_servers(fingerprint, servers, expected):
    n, explanation, _ = deduced_gpus(config(fingerprint, servers))
    assert n == expected and explanation.startswith(f"{expected} (TP ")


def test_explanation_lists_every_factor():
    n, explanation, masked = deduced_gpus(config("vllm-0.11.0-tp2-pp3-dp2-0a1b2c3d", servers=2))
    assert (n, masked) == (24, False)
    assert explanation == "24 (TP 2 × PP 3 × DP 2 × 2 vLLM server(s))"


def test_masking_possible_when_the_fingerprint_has_no_degree():
    assert deduced_gpus(config("vllm-0.11.0-0a1b2c3d"))[2] is True
    assert deduced_gpus(config("vllm-0.11.0-tp2-0a1b2c3d"))[2] is False
    assert deduced_gpus(config("vllm-0.11.0-ep-0a1b2c3d"))[2] is False     # the -ep segment is a degree segment


def test_prefill_context_parallel_size_multiplies_when_server_info_was_read():
    n, explanation, _ = deduced_gpus(config("vllm-0.11.0-tp4-0a1b2c3d", servers=2, pcp=2))
    assert n == 16 and explanation == "16 (TP 4 × PP 1 × DP 1 × PCP 2 × 2 vLLM server(s))"


@pytest.mark.parametrize("pcp", [None, 0, 1])
def test_pcp_of_one_or_unknown_is_not_shown(pcp):
    n, explanation, _ = deduced_gpus(config("vllm-0.11.0-tp4-0a1b2c3d", pcp=pcp))
    assert n == 4 and "PCP" not in explanation


def test_pcp_is_read_from_the_first_vllm_server_only():
    r = config("vllm-0.11.0-tp2-0a1b2c3d", servers=2)
    r["servers"][1]["configuration"] = {"parallelism": {"prefill_context_parallel_size": 4}}
    assert deduced_gpus(r)[0] == 4
    r["servers"][0]["vllm_lines"] = 0         # the first server is not vLLM's: the second one answers
    assert deduced_gpus(r)[0] == 8


def test_servers_that_are_not_vllm_are_not_counted():
    r = config("vllm-0.11.0-tp2-0a1b2c3d", servers=3)
    r["servers"][1]["vllm_lines"] = 0          # LiteLLM address given by mistake
    r["servers"][2]["vllm_lines"] = None       # unreachable
    n, explanation, _ = deduced_gpus(r)
    assert n == 2 and explanation.endswith("× 1 vLLM server(s))")


@pytest.mark.parametrize("fingerprint", [None, {}, {"raw": None, "response": False, "decoded": None},
                                         {"raw": "x", "decoded": None}])
def test_no_fingerprint_no_count(fingerprint):
    r = config()
    r["fingerprint"] = fingerprint
    n, explanation, masked = deduced_gpus(r)
    assert n is None and masked is False and explanation.startswith("fingerprint absent")


def test_no_vllm_server_no_count():
    for servers in ([], None, [config(vllm_lines=0)["servers"][0]], [config(vllm_lines=None, metrics_status=401)["servers"][0]]):
        r = config()
        r["servers"] = servers
        n, explanation, _ = deduced_gpus(r)
        assert n is None and explanation == "no metrics URL answers like a vLLM server (BENCH_METRICS empty or wrong)"
    r = config()
    del r["servers"]
    assert deduced_gpus(r)[0] is None


def test_old_files_without_vllm_lines_recognise_a_vllm_server_by_what_was_read():
    base = config()
    for old, ok in (({"speculative_tokens_metrics": 3}, True), ({"speculative_tokens_metrics": 0}, True),
                    ({"cache_config": {"block_size": "16"}}, True), ({"cache_config": {}}, False),
                    ({"version": "0.11.0"}, False), ({}, False)):
        entry = {"metrics": "m", "version": "0.11.0", "cache_config": None, "configuration": None,
                 "speculative_tokens_metrics": None, **old}
        r = {**base, "servers": [entry]}
        assert (deduced_gpus(r)[0] == 8) is ok, old


def test_vllm_lines_wins_over_the_old_markers():
    r = config(vllm_lines=0, speculative_tokens_metrics=3, cache_config={"block_size": "16"})
    assert deduced_gpus(r)[0] is None


@pytest.mark.parametrize("broken", [None, [], "text", 5,
                                    {"fingerprint": {"decoded": {"tensor_parallel_size": 2}}, "servers": [{"vllm_lines": 1}]},
                                    {"fingerprint": {"decoded": {"tensor_parallel_size": "x", "pipeline_parallel_size": 1,
                                                                  "data_parallel_size": 1}}, "servers": [{"vllm_lines": 1}]},
                                    {"fingerprint": {"decoded": {"tensor_parallel_size": None, "pipeline_parallel_size": 1,
                                                                  "data_parallel_size": 1}}, "servers": [{"vllm_lines": 1}]},
                                    {"fingerprint": "text", "servers": []},
                                    {"fingerprint": {"decoded": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1,
                                                                  "data_parallel_size": 1}}, "servers": [5]}])
def test_unexpected_shapes_give_a_reason_and_no_exception(broken):
    n, explanation, masked = deduced_gpus(broken)
    assert n is None and masked is False
    assert explanation.startswith("server_config.json has an unexpected shape (") or explanation.startswith("fingerprint absent")


# ---------------------------------------------------------------- run_gpus
def test_run_gpus_reads_the_file_of_a_run_directory(tmp_path):
    (tmp_path / "server_config.json").write_text(json.dumps(config()))
    assert run_gpus(tmp_path) == deduced_gpus(config()) and run_gpus(str(tmp_path))[0] == 8


def test_run_gpus_missing_file(tmp_path):
    assert run_gpus(tmp_path) == (None, "server_config.json missing", False)
    assert run_gpus(tmp_path / "no-such-run")[1] == "server_config.json missing"


@pytest.mark.parametrize("content", [b"{not json", b"", b"\xff\xfe\x00bad"])
def test_run_gpus_unreadable_file(tmp_path, content):
    (tmp_path / "server_config.json").write_bytes(content)
    n, reason, masked = run_gpus(tmp_path)
    assert n is None and masked is False and reason.startswith("server_config.json unreadable (")


def test_run_gpus_a_directory_in_place_of_the_file(tmp_path):
    (tmp_path / "server_config.json").mkdir()
    assert run_gpus(tmp_path)[1].startswith("server_config.json unreadable (")


def test_run_gpus_json_of_the_wrong_type(tmp_path):
    (tmp_path / "server_config.json").write_text("[1, 2]")
    assert run_gpus(tmp_path)[1].startswith("server_config.json has an unexpected shape (")


def test_run_gpus_file_without_the_new_keys(tmp_path):
    old = config()
    for entry in old["servers"]:
        del entry["vllm_lines"], entry["metrics_status"]
    old["servers"][0]["speculative_tokens_metrics"] = 3
    (tmp_path / "server_config.json").write_text(json.dumps(old))
    assert run_gpus(tmp_path)[0] == 8


# ---------------------------------------------------------------- GpuResolver
def run_dir(tmp_path, name, cfg=None):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        (d / "server_config.json").write_text(json.dumps(cfg))
    return d


@pytest.mark.parametrize("given", [0, -1, -8])
def test_resolver_rejects_a_gpu_below_one(given):
    with pytest.raises(SystemExit) as e:
        GpuResolver(given)
    assert str(e.value) == "--gpu: integer >= 1"


def test_resolver_given_value_wins_and_is_returned_even_without_a_file(tmp_path):
    resolver = GpuResolver(3)
    assert resolver.for_run(run_dir(tmp_path, "r1")) == 3
    assert resolver.notes == {} and resolver.unknown == {}


def test_resolver_warns_when_the_given_value_contradicts_the_fingerprint(tmp_path, capsys):
    resolver = GpuResolver(4)
    assert resolver.for_run(run_dir(tmp_path, "r1", config())) == 4
    resolver.finish([])
    assert capsys.readouterr().out == ("WARNING: r1: --gpu 4, but the fingerprint gives 8 (TP 4 × PP 1 × DP 2 × 1 vLLM server(s)): "
                                       "check the value.\n")


def test_resolver_no_warning_when_the_values_agree_or_the_fingerprint_may_be_masked(tmp_path, capsys):
    resolver = GpuResolver(8)
    resolver.for_run(run_dir(tmp_path, "agree", config()))
    resolver = GpuResolver(8)
    resolver.for_run(run_dir(tmp_path, "masked", config("vllm-0.11.0-0a1b2c3d")))      # deduced 1, but may be hidden
    resolver.for_run(run_dir(tmp_path, "nothing", config(servers=0)))
    resolver.finish([])
    assert capsys.readouterr().out == "" and resolver.notes == {}


def test_resolver_deduces_when_nothing_is_given(tmp_path, capsys):
    resolver = GpuResolver(None)
    assert resolver.for_run(run_dir(tmp_path, "conf-r1", config())) == 8
    resolver.finish([])
    assert capsys.readouterr().out == "GPUs: conf-r1: 8 (TP 4 × PP 1 × DP 2 × 1 vLLM server(s)), deduced from the fingerprint.\n"


def test_resolver_masked_fingerprint_note_warns_about_one_gpu_per_server(tmp_path, capsys):
    resolver = GpuResolver(None)
    assert resolver.for_run(run_dir(tmp_path, "m", config("vllm-0.11.0-0a1b2c3d"))) == 1
    resolver.finish([])
    out = capsys.readouterr().out
    assert out.startswith("GPUs: m: 1 (TP 1 × PP 1 × DP 1 × 1 vLLM server(s)), deduced from the fingerprint; WARNING: 1 GPU per server")
    assert "--fingerprint-mode hash would hide it): give --gpu if the server has more." in out


def test_resolver_unknown_runs_stop_with_the_reasons_sorted_by_run(tmp_path):
    resolver = GpuResolver(None)
    assert resolver.for_run(run_dir(tmp_path, "b-r1")) is None
    assert resolver.for_run(run_dir(tmp_path, "a-r1", config(servers=0))) is None
    assert resolver.for_run(run_dir(tmp_path, "c-r1", config())) == 8
    with pytest.raises(SystemExit) as e:
        resolver.finish([])
    assert str(e.value) == ("ERROR: number of GPUs unknown for a-r1 (no metrics URL answers like a vLLM server (BENCH_METRICS empty or wrong)); "
                            "b-r1 (server_config.json missing): give --gpu (\"GPU\" field in the interface).")


def test_resolver_prints_nothing_before_finishing(tmp_path, capsys):
    resolver = GpuResolver(None)
    resolver.for_run(run_dir(tmp_path, "r1", config()))
    assert capsys.readouterr().out == ""


def test_resolver_repetitions_with_different_gpu_counts_are_flagged(capsys):
    rows = [{"run": "prod-r1", "gpu": 8}, {"run": "prod-r2", "gpu": 4}, {"run": "prod-r3", "gpu": 8},
            {"run": "other-r1", "gpu": 2}, {"run": "other-r2", "gpu": 2}, {"run": "solo", "gpu": 1}]
    GpuResolver(None).finish(rows)
    assert capsys.readouterr().out == ("WARNING: prod: number of GPUs differs between repetitions (prod-r1 = 8, prod-r2 = 4, prod-r3 = 8): "
                                       "the median mixes configurations; check, or give --gpu.\n")


def test_resolver_repetition_check_skips_rows_without_a_gpu_value(capsys):
    GpuResolver(None).finish([{"run": "a-r1", "gpu": 8}, {"run": "a-r2"}, {"run": "a-r3", "gpu": None}, {"run": "a-r1", "gpu": 8}])
    assert capsys.readouterr().out == ""


def test_resolver_notes_precede_the_repetition_warnings(tmp_path, capsys):
    resolver = GpuResolver(None)
    resolver.for_run(run_dir(tmp_path, "p-r1", config("vllm-0.11.0-tp8-0a1b2c3d")))
    resolver.for_run(run_dir(tmp_path, "p-r2", config("vllm-0.11.0-tp4-0a1b2c3d")))
    resolver.finish([{"run": "p-r1", "gpu": 8}, {"run": "p-r2", "gpu": 4}])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("GPUs: p-r1: 8") and lines[1].startswith("GPUs: p-r2: 4") and lines[2].startswith("WARNING: p:")


def test_common_has_no_other_gpu_entry_point():
    assert hasattr(common, "GpuResolver")


# ---------------------------------------------------------------- the reports
def with_configs(root, **cfgs):
    for run, cfg in cfgs.items():
        run_dir(root, run.replace("_", "-"), cfg)


def table(root, name):
    import csv
    with open(root / name, newline="") as fh:
        return list(csv.DictReader(fh))


def test_aiperf_without_gpu_deduces_it_per_run(tmp_path, capsys):
    root = fixtures.aiperf_tree(tmp_path)
    with_configs(root, conf_a_r1=config("vllm-1-tp4-dp2-0a1b2c3d"), conf_a_r2=config("vllm-1-tp4-dp2-0a1b2c3d"),
                 conf_b_r1=config("vllm-1-tp2-0a1b2c3d"), conf_c_r1=config("vllm-1-tp2-0a1b2c3d", servers=2))
    main(["aiperf", str(root), "--target", "30"])
    out = capsys.readouterr().out
    assert out.startswith("GPUs: conf-a-r1: 8 (TP 4 × PP 1 × DP 2 × 1 vLLM server(s)), deduced from the fingerprint.\n")
    assert "GPUs: conf-b-r1: 2 (TP 2 × PP 1 × DP 1 × 1 vLLM server(s)), deduced from the fingerprint.\n" in out
    assert "GPUs: conf-c-r1: 4 (TP 2 × PP 1 × DP 1 × 2 vLLM server(s)), deduced from the fingerprint.\n" in out
    rows = table(root, "table.csv")
    assert {r["run"]: r["gpu"] for r in rows} == {"conf-a-r1": "8", "conf-a-r2": "8", "conf-b-r1": "2", "conf-c-r1": "4"}
    a1 = [r for r in rows if r["run"] == "conf-a-r1" and r["concurrency"] == "4"][0]
    assert float(a1["output_throughput_per_gpu"]) == pytest.approx(float(a1["total_output_throughput"]) / 8)
    assert list(rows[0]).index("gpu") == list(rows[0]).index("total_output_throughput") + 1


def test_aiperf_gpu_option_overrides_and_warns_on_contradiction(tmp_path, capsys):
    root = fixtures.aiperf_tree(tmp_path)
    with_configs(root, conf_a_r1=config("vllm-1-tp4-dp2-0a1b2c3d"), conf_a_r2=config("vllm-1-tp2-0a1b2c3d"))
    main(["aiperf", str(root), "--target", "30", "--gpu", "2"])
    out = capsys.readouterr().out
    assert "WARNING: conf-a-r1: --gpu 2, but the fingerprint gives 8 (TP 4 × PP 1 × DP 2 × 1 vLLM server(s)): check the value." in out
    assert "conf-a-r2: --gpu 2, but" not in out and "deduced from the fingerprint" not in out
    assert {r["gpu"] for r in table(root, "table.csv")} == {"2"}


def test_aiperf_warns_when_repetitions_have_different_gpu_counts(tmp_path, capsys):
    root = fixtures.aiperf_tree(tmp_path)
    with_configs(root, conf_a_r1=config("vllm-1-tp8-0a1b2c3d"), conf_a_r2=config("vllm-1-tp4-0a1b2c3d"),
                 conf_b_r1=config("vllm-1-tp4-0a1b2c3d"), conf_c_r1=config("vllm-1-tp4-0a1b2c3d"))
    main(["aiperf", str(root), "--target", "30"])
    assert ("WARNING: conf-a: number of GPUs differs between repetitions (conf-a-r1 = 8, conf-a-r2 = 4): "
            "the median mixes configurations; check, or give --gpu.") in capsys.readouterr().out


def test_aiperf_unknown_gpu_stops_before_writing_anything(tmp_path, capsys):
    root = fixtures.aiperf_tree(tmp_path)
    with_configs(root, conf_a_r1=config("vllm-1-tp4-0a1b2c3d"), conf_a_r2=config("vllm-1-tp4-0a1b2c3d"))
    (root / "conf-b-r1" / "server_config.json").write_text("{broken")
    with pytest.raises(SystemExit) as e:
        main(["aiperf", str(root), "--target", "30"])
    message = str(e.value)
    assert message.startswith("ERROR: number of GPUs unknown for conf-b-r1 (server_config.json unreadable (")
    assert "conf-c-r1 (server_config.json missing)" in message and message.endswith('give --gpu ("GPU" field in the interface).')
    assert not (root / "table.csv").exists() and capsys.readouterr().out == ""


def test_aiperf_gpu_given_makes_missing_files_harmless(tmp_path, capsys):
    root = fixtures.aiperf_tree(tmp_path)
    main(["aiperf", str(root), "--target", "30", "--gpu", "1"])
    out = capsys.readouterr().out
    assert "GPUs:" not in out and "WARNING: conf" not in out.split("Table:")[0]
    assert {r["gpu"] for r in table(root, "table.csv")} == {"1"}


@pytest.mark.parametrize("kind,root_of", [("aiperf", fixtures.aiperf_tree), ("agents", fixtures.agents_tree)])
def test_reports_reject_gpu_below_one(tmp_path, kind, root_of):
    root = root_of(tmp_path)
    with pytest.raises(SystemExit) as e:
        main([kind, str(root), "--target", "30", "--gpu", "0"])
    assert str(e.value) == "--gpu: integer >= 1"


def test_agents_without_gpu_deduces_it_and_fills_the_all_rows(tmp_path, capsys):
    root = fixtures.agents_tree(tmp_path)
    with_configs(root, conf_r1=config("vllm-1-tp4-0a1b2c3d", servers=2))
    main(["agents", str(root), "--target", "30"])
    out = capsys.readouterr().out
    assert out.startswith("GPUs: conf-r1: 8 (TP 4 × PP 1 × DP 1 × 2 vLLM server(s)), deduced from the fingerprint.\n")
    rows = table(root, "table_agents.csv")
    assert [(r["agent"], r["gpu"]) for r in rows if r["concurrent_tasks"] == "1"] == [("all", "8"), ("code_review", ""), ("pentest", "")]
    assert list(rows[0]).index("gpu") == list(rows[0]).index("max_context") + 1
    all_row = [r for r in rows if r["agent"] == "all" and r["concurrent_tasks"] == "1"][0]
    assert float(all_row["output_throughput_per_gpu"]) == pytest.approx(9.2 / 8)


def test_agents_contradiction_and_unknown(tmp_path, capsys):
    root = fixtures.agents_tree(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["agents", str(root), "--target", "30"])
    assert str(e.value).startswith("ERROR: number of GPUs unknown for conf-r1 (server_config.json missing)")
    assert not (root / "table_agents.csv").exists()
    with_configs(root, conf_r1=config("vllm-1-tp4-0a1b2c3d"))
    main(["agents", str(root), "--target", "30", "--gpu", "1"])
    assert "WARNING: conf-r1: --gpu 1, but the fingerprint gives 4 (TP 4 × PP 1 × DP 1 × 1 vLLM server(s)): check the value." in capsys.readouterr().out


def test_agents_repetitions_with_different_gpu_counts(tmp_path, capsys):
    root = fixtures.agents_tree(tmp_path)
    (root / "conf-r1").rename(root / "conf-r1-r1")
    import shutil
    shutil.copytree(root / "conf-r1-r1", root / "conf-r1-r2")
    with_configs(root, conf_r1_r1=config("vllm-1-tp4-0a1b2c3d"), conf_r1_r2=config("vllm-1-tp2-0a1b2c3d"))
    main(["agents", str(root), "--target", "30"])
    assert "WARNING: conf-r1: number of GPUs differs between repetitions (conf-r1-r1 = 4, conf-r1-r2 = 2)" in capsys.readouterr().out
