"""Read the bench's environment settings. Only BENCH_<NAME> variables are read.

Names: BENCH_URL, BENCH_MODEL, BENCH_METRICS, BENCH_API_KEY, BENCH_TOKENIZER, BENCH_LITELLM_KEY,
BENCH_AGENT_KEYS (and, for the shell scripts, BENCH_ISL, BENCH_OSL, BENCH_DURATION, ...).
Variables from older versions or with generic names (API_KEY, URL, ...) are NOT read, because a generic
name exported for another tool must never be sent to the target. `warn_ignored_settings` tells the user
when such a variable is set (it never prints values). Run `python3 tools/settings.py` to see the warnings."""
import os
import sys

# variable that is no longer read -> variable to use instead
IGNORED = {
    "URL": "BENCH_URL", "MODEL": "BENCH_MODEL", "METRICS": "BENCH_METRICS", "API_KEY": "BENCH_API_KEY",
    "TOKENIZER": "BENCH_TOKENIZER", "MODELE": "BENCH_MODEL", "METRIQUES": "BENCH_METRICS", "CLE_API": "BENCH_API_KEY",
    "BANC_URL": "BENCH_URL", "BANC_MODELE": "BENCH_MODEL", "BANC_METRIQUES": "BENCH_METRICS",
    "BANC_CLE_API": "BENCH_API_KEY", "BANC_TOKENIZER": "BENCH_TOKENIZER", "BANC_CLE_LITELLM": "BENCH_LITELLM_KEY",
    "BANC_CLES_AGENTS": "BENCH_AGENT_KEYS", "BANC_JETON": "BENCH_TOKEN",
    "ISL": "BENCH_ISL", "OSL": "BENCH_OSL", "DUREE": "BENCH_DURATION", "DURATION": "BENCH_DURATION",
    "TEMPERATURE": "BENCH_TEMPERATURE", "RACINE": "BENCH_ROOT_DIR", "ROOT_DIR": "BENCH_ROOT_DIR",
    "ECRASER": "BENCH_OVERWRITE", "OVERWRITE": "BENCH_OVERWRITE", "EXTRA": "BENCH_EXTRA", "SEED": "BENCH_SEED",
    "REQ_PAR_CONC": "BENCH_REQUESTS_PER_CONCURRENCY", "REQUESTS_PER_CONCURRENCY": "BENCH_REQUESTS_PER_CONCURRENCY",
    "CONTEXTE_MAX": "BENCH_MAX_CONTEXT", "MAX_CONTEXT": "BENCH_MAX_CONTEXT", "ESSAI_COURT": "BENCH_SHORT_RUN",
    "SHORT_RUN": "BENCH_SHORT_RUN", "AGENTX_DOSSIER": "BENCH_AGENTX_DIR", "AGENTX_DIR": "BENCH_AGENTX_DIR",
    "AGENTX_VARIANTE": "BENCH_AGENTX_VARIANT", "AGENTX_VARIANT": "BENCH_AGENTX_VARIANT",
}


def read_setting(name, default=None):
    """Value of BENCH_<name>, else `default` (an empty value counts as absent)."""
    return os.environ.get("BENCH_" + name) or default


def api_key():
    """The target's API key (BENCH_API_KEY), or None."""
    return read_setting("API_KEY")


def warn_ignored_settings(stream=None):
    """Warn (names only, never values) about set variables that are no longer read; return how many."""
    stream = stream or sys.stderr
    count = 0
    for old, new in IGNORED.items():
        if os.environ.get(old) and not os.environ.get(new):
            print(f"WARNING: {old} is set but is no longer read; use {new} instead.", file=stream)
            count += 1
    return count


if __name__ == "__main__":
    warn_ignored_settings()
