"""`mlxserve opencode`: launch OpenCode wired to this server, no config file to write.

The OpenCode config is passed through OPENCODE_CONFIG_CONTENT (merged over the user's global
config, which stays untouched). The server is started in the background if it is not running.
MCP servers from the user's config are disabled for the session (`--keep-mcp` keeps them): a local
model cannot take hundreds of tool schemas (one MCP server alone sent 217 tools / 814 KB per request).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from mlxserve.config import settings

LOG_PATH = Path.home() / "Library" / "Logs" / "mlxserve.log"


def base_url() -> str:
    host = "127.0.0.1" if settings.host in ("0.0.0.0", "::", "") else settings.host
    return f"http://{host}:{settings.port}"


def _get(url: str, timeout: float = 2.0) -> dict | None:
    req = urllib.request.Request(url)
    if settings.api_key:
        req.add_header("Authorization", f"Bearer {settings.api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def ensure_server(url: str, wait_s: float = 60.0) -> bool:
    if _get(f"{url}/health"):
        return True
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting MLXServe in the background (logs: {LOG_PATH})…", file=sys.stderr)
    with LOG_PATH.open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "mlxserve.cli", "serve"],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if _get(f"{url}/health"):
            return True
        time.sleep(0.5)
    return False


def user_mcp_servers(exe: str) -> list[str]:
    """MCP servers in the resolved OpenCode config (global + project) for the current directory."""
    try:
        out = subprocess.run([exe, "debug", "config"], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30)
        return list(json.loads(out.stdout).get("mcp") or {})
    except Exception:
        return []


def opencode_config(url: str, model: dict, disable_mcp: list[str] = ()) -> dict:
    model_id = model["id"]
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "mlxserve": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "MLXServe (local)",
                "options": {"baseURL": f"{url}/v1", "apiKey": settings.api_key or "local"},
                "models": {
                    model_id: {
                        "name": f"{model_id.split('/')[-1]} (MLXServe)",
                        "tool_call": True,
                        "limit": {
                            "context": model.get("context_length") or settings.context_tokens,
                            "output": settings.max_output_tokens,
                        },
                    }
                },
            }
        },
        "model": f"mlxserve/{model_id}",
        "small_model": f"mlxserve/{model_id}",
        "mcp": {name: {"enabled": False} for name in disable_mcp},
    }


def run(args: list[str]) -> int:
    exe = shutil.which("opencode")
    if not exe:
        print("OpenCode is not installed: brew install sst/tap/opencode  (or: npm i -g opencode-ai)", file=sys.stderr)
        return 1
    url = base_url()
    if not ensure_server(url):
        print(f"MLXServe did not start, see {LOG_PATH}", file=sys.stderr)
        return 1
    models = (_get(f"{url}/v1/models") or {}).get("data") or []
    if not models:
        print(f"No model served by {url}/v1/models", file=sys.stderr)
        return 1
    keep_mcp = "--keep-mcp" in args
    args = [a for a in args if a != "--keep-mcp"]
    disable = [] if keep_mcp else user_mcp_servers(exe)
    if disable:
        print(f"MCP servers disabled for this session (--keep-mcp to keep): {', '.join(disable)}", file=sys.stderr)
    env = dict(os.environ, OPENCODE_CONFIG_CONTENT=json.dumps(opencode_config(url, models[0], disable)))
    os.execvpe(exe, [exe, *args], env)
