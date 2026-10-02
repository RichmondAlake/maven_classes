"""Launch on loopback with mandatory raw Anthropic and Tavily clients."""
import argparse
import getpass
import os
from pathlib import Path
from dotenv import dotenv_values
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8031)
parser.add_argument(
    "--memory-backend", choices=["scratch", "memorizz"], default="memorizz"
)
parser.add_argument("--env-file", type=Path)
parser.add_argument("--ask-keys", action="store_true")
args = parser.parse_args()
private = dotenv_values(args.env_file) if args.env_file else {}
for name in ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]:
    value = None if args.ask_keys else os.getenv(name) or private.get(name)
    if not value:
        value = getpass.getpass(f"{name} (hidden, process memory only): ").strip()
    if not value:
        raise ValueError(f"{name} is required.")
    os.environ[name] = value
if private.get("TYPESAFE_API_KEY"):
    os.environ.setdefault("TYPESAFE_API_KEY", private["TYPESAFE_API_KEY"])
os.environ.update(MEMORY_BACKEND=args.memory_backend, NOTEBOOK_USE_ENV_KEYS="1")
os.environ.setdefault("ANTHROPIC_MODEL", "claude-opus-5-5")
os.environ.setdefault("HF_HOME", "/private/tmp/maven_hf_cache")
uvicorn.run("backend.main:app", host="127.0.0.1", port=args.port, access_log=False)
