"""docker-compose.yml: what must stay true without running Docker (read as text, no YAML library needed)."""
import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def test_every_published_port_is_on_the_loopback_of_the_host_only():
    published = re.findall(r'ports:\s*\["([^"]+)"\]', COMPOSE)
    assert len(published) == 3                       # the interface, Prometheus, LiteLLM
    assert all(p.startswith("127.0.0.1:") for p in published), published


def test_the_optional_services_need_a_profile():
    for service in ("litellm", "litellm-db", "fake-server"):
        block = COMPOSE.split(f"\n  {service}:")[1].split("\n\n")[0]
        assert "profiles:" in block, service
    for service in ("bench", "prometheus"):
        assert "profiles:" not in COMPOSE.split(f"\n  {service}:")[1].split("\n\n")[0], service


def test_the_prometheus_configurations_exist_and_keys_stay_out_of_git():
    for name in ("prometheus.yml", "prometheus-gateway.yml", "prometheus-fake.yml"):
        assert (ROOT / "docker" / name).is_file(), name
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docker/secrets/*" in ignored and "!docker/secrets/fake.key" in ignored
    assert (ROOT / "docker" / "secrets" / "fake.key").read_text() == "fake"      # a placeholder, not a secret


def test_the_interface_is_told_to_listen_inside_the_container_only():
    command = re.search(r'command: \["python", "-m", "interface.server"[^\]]*\]', COMPOSE).group(0)
    assert '"--host", "0.0.0.0"' in command and "--public-port" in command


def test_the_litellm_config_has_the_settings_the_per_agent_view_needs():
    config = (ROOT / "docker" / "litellm-config.yaml").read_text(encoding="utf-8")
    assert 'callbacks: ["prometheus"]' in config and "metadata.agent" in config
    assert "always_include_stream_usage: true" in config


# ---------------------------------------------------------------- the tokenizer downloaded once at start
def run_entrypoint(tmp_path, env, has_tokenizer=False, download_ok=True):
    """The real docker/entrypoint.sh in a scratch folder with a stand-in for the download script."""
    import subprocess
    (tmp_path / "tools" / "tokenizer").mkdir(parents=True)
    if has_tokenizer:
        (tmp_path / "tools" / "tokenizer" / "tokenizer.json").write_text("{}")
        (tmp_path / "tools" / "tokenizer" / "tokenizer_config.json").write_text("{}")
    stub = tmp_path / "tools" / "download_tokenizer.sh"
    stub.write_text(f'#!/bin/sh\necho "DOWNLOAD $1 ${{HF_TOKEN:-}}" >> calls.txt\n{"exit 0" if download_ok else "exit 1"}\n')
    stub.chmod(0o755)
    out = subprocess.run([shutil.which("bash"), str(ROOT / "docker" / "entrypoint.sh"), "sh", "-c", 'echo "STARTED token=[${HF_TOKEN:-}]"'],
                         cwd=tmp_path, env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True, timeout=30)
    calls = (tmp_path / "calls.txt").read_text().split("\n")[:-1] if (tmp_path / "calls.txt").exists() else []
    return out, calls


def test_the_tokenizer_is_downloaded_once_then_the_command_runs(tmp_path):
    out, calls = run_entrypoint(tmp_path, {"BENCH_TOKENIZER_REPO": "org/model"})
    assert calls == ["DOWNLOAD org/model "] and "STARTED" in out.stdout and out.returncode == 0


def test_no_download_when_a_tokenizer_is_there_or_no_repo_is_given(tmp_path):
    out, calls = run_entrypoint(tmp_path / "a", {"BENCH_TOKENIZER_REPO": "org/model"}, has_tokenizer=True)
    assert calls == [] and "STARTED" in out.stdout
    out, calls = run_entrypoint(tmp_path / "b", {})
    assert calls == [] and "STARTED" in out.stdout


def test_a_failed_download_warns_and_the_command_still_runs(tmp_path):
    out, calls = run_entrypoint(tmp_path, {"BENCH_TOKENIZER_REPO": "org/model"}, download_ok=False)
    assert calls == ["DOWNLOAD org/model "] and "WARNING: the tokenizer download failed" in out.stdout
    assert "STARTED" in out.stdout and out.returncode == 0


def test_litellm_has_its_database_and_waits_for_it():
    litellm = COMPOSE.split("\n  litellm:")[1].split("\n\n")[0]
    database = COMPOSE.split("\n  litellm-db:")[1].split("\n\n")[0]
    assert "DATABASE_URL" in litellm and "service_healthy" in litellm
    assert "healthcheck:" in database and "ports:" not in database          # no port is published
    assert 'POSTGRES_PASSWORD: "${LITELLM_DB_PASSWORD:-}"' in database       # empty: PostgreSQL refuses to start
    assert "litellm-db-data" in COMPOSE.split("\nvolumes:")[-1]


def test_the_token_is_used_for_the_download_only(tmp_path):
    out, calls = run_entrypoint(tmp_path, {"BENCH_TOKENIZER_REPO": "org/model", "HF_TOKEN": "hf_secret"})
    assert calls == ["DOWNLOAD org/model hf_secret"]                    # the download had it
    assert "hf_secret" not in out.stdout and "token=[]" in out.stdout   # the command does not


def test_a_partial_tokenizer_is_downloaded_again(tmp_path):
    import subprocess
    (tmp_path / "tools" / "tokenizer").mkdir(parents=True)
    (tmp_path / "tools" / "tokenizer" / "tokenizer.json").write_text("{}")       # the config file is missing
    stub = tmp_path / "tools" / "download_tokenizer.sh"
    stub.write_text('#!/bin/sh\necho "DOWNLOAD $1" >> calls.txt\n')
    stub.chmod(0o755)
    subprocess.run([shutil.which("bash"), str(ROOT / "docker" / "entrypoint.sh"), "true"], cwd=tmp_path,
                   env={"PATH": "/usr/bin:/bin", "BENCH_TOKENIZER_REPO": "org/model"}, capture_output=True, timeout=30)
    assert (tmp_path / "calls.txt").read_text() == "DOWNLOAD org/model\n"


def test_prometheus_gets_litellms_key_from_the_master_key_without_a_manual_step():
    key = COMPOSE.split("\n  prometheus-key:")[1].split("\n\n")[0]
    assert "profiles: [gateway]" in key and "LITELLM_MASTER_KEY" in key and "litellm-key:/keys" in key
    prometheus = COMPOSE.split("\n  prometheus:")[1].split("\n\n")[0]
    assert "litellm-key:/etc/prometheus/litellm:ro" in prometheus
    assert "service_completed_successfully" in prometheus and "required: false" in prometheus
    config = (ROOT / "docker" / "prometheus-gateway.yml").read_text(encoding="utf-8")
    assert "credentials_file: /etc/prometheus/litellm/litellm.key" in config
    assert "litellm-key:" in COMPOSE.split("\nvolumes:")[-1]


def test_the_connection_comes_from_dot_env_to_the_bench_container():
    bench = COMPOSE.split("\n  bench:")[1].split("\n\n")[0]
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for name in ("BENCH_URL", "BENCH_MODEL", "BENCH_METRICS", "BENCH_API_KEY", "BENCH_TOKENIZER"):
        assert f'{name}: "${{{name}:-}}"' in bench, name       # empty when unset: the scripts' defaults apply
        assert f"export {name}=" in example, name
