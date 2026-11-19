"""docker-compose.yml: what must stay true without running Docker (read as text, no YAML library needed)."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def test_every_published_port_is_on_the_loopback_of_the_host_only():
    published = re.findall(r'ports:\s*\["([^"]+)"\]', COMPOSE)
    assert len(published) == 3                       # the interface, Prometheus, LiteLLM
    assert all(p.startswith("127.0.0.1:") for p in published), published


def test_the_optional_services_need_a_profile():
    for service in ("litellm", "fake-server"):
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
