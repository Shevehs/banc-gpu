"""Helpers for the tests: free ports, a stub HTTP server, a subprocess runner and the fake server."""
import contextlib
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent


def free_port():
    """A TCP port on 127.0.0.1 that is free right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def have(module):
    """True if the module can be imported (without importing it)."""
    return importlib.util.find_spec(module) is not None


def clean_environment(**extra):
    """Environment for a subprocess: no bench setting, no bytecode files, no proxy; plus `extra`."""
    from tools.settings import IGNORED
    env = {k: v for k, v in os.environ.items() if not k.startswith("BENCH_") and k not in IGNORED}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env.update(extra)
    return env


def run(args, cwd=ROOT, env=None, timeout=120, **kw):
    """subprocess.run with captured text output and a clean environment."""
    return subprocess.run(args, cwd=str(cwd), env=env if env is not None else clean_environment(),
                          capture_output=True, text=True, timeout=timeout, **kw)


class Stub:
    """A local HTTP server. routes: path -> (status, body or callable returning it, headers). Requests go to `seen`."""

    def __init__(self, routes):
        self.routes, self.seen = dict(routes), []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                stub.seen.append({"method": method, "path": self.path, "headers": dict(self.headers), "body": body})
                status, payload, headers = stub.routes.get(self.path.split("?")[0], (404, "", {}))
                payload = payload() if callable(payload) else payload
                data = json.dumps(payload).encode() if isinstance(payload, (dict, list)) else str(payload).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._answer("GET")

            def do_POST(self):
                self._answer("POST")

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class PromStub:
    """A local Prometheus HTTP API: GET /api/v1/query_range is answered by `responder(query, start, end, step)`,
    which returns the Prometheus "result" list ([{"metric": {...}, "values": [[t, "v"], ...]}]). Calls go to `seen`."""

    def __init__(self, responder):
        self.responder, self.seen = responder, []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = urllib.parse.urlparse(self.path)
                qs = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
                if url.path != "/api/v1/query_range":
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                call = {"query": qs["query"], "start": float(qs["start"]), "end": float(qs["end"]),
                        "step": float(qs["step"]), "headers": dict(self.headers)}
                stub.seen.append(call)
                body = json.dumps({"status": "success", "data": {"resultType": "matrix", "result": stub.responder(
                    call["query"], call["start"], call["end"], call["step"])}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@contextlib.contextmanager
def fake_server(*args):
    """Run tools/fake_server.py on a free port; yields its base URL. Stops it afterwards."""
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-B", str(ROOT / "tools" / "fake_server.py"), "--port", str(port), *args],
                            cwd=str(ROOT), env=clean_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 20
        while True:
            if proc.poll() is not None:
                raise RuntimeError("fake server exited: " + proc.stdout.read().decode(errors="replace"))
            try:
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + "/version", timeout=1).read()
                break
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError("fake server did not start") from None
                time.sleep(0.1)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        proc.stdout.close()
