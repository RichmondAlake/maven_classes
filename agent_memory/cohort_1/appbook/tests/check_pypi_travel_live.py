"""Exercise published-package travel research in an isolated appbook user."""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
args = parser.parse_args()
BASE = os.getenv("APPBOOK_URL", "http://127.0.0.1:8031")
ROOT = Path(__file__).resolve().parents[1]
checks = []


def call(path, payload=None):
    response = requests.get(BASE + path, timeout=300) if payload is None else requests.post(BASE + path, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def check(name, condition):
    assert condition, name
    checks.append(name)
    print("PASS:", name, flush=True)


deadline = time.monotonic() + 180
while not call("/api/status")["ready"]:
    if time.monotonic() > deadline:
        raise RuntimeError("Appbook did not finish warming up")
    time.sleep(1)

original = call("/api/state")
try:
    fresh = call("/api/users", {"display_name": "Package validation"})
    call("/api/trip", {"request": args.request})
    result = call("/api/chat", {"message": args.request + " Keep the research shortlist concise."})
    check("Published-package app produces a sourced live travel answer", bool(result["answer"]) and bool(result["sources"]))
    check("Each decision receives freshly loaded workflow memory", all("workflow_memory" in step["context"] for step in result["inspections"]) and any(step["context"]["workflow_memory"] for step in result["inspections"][1:]))
    check("Anthropic reports cache reads during the real agent loop", sum(row["cache_read_tokens"] for row in result["calls"]) > 0)
    context = call("/api/context")
    check("Context inspector exposes a bounded dynamic memory suffix", context["prefix"][0]["cache_control"]["type"] == "ephemeral" and context["budget"]["after_tokens"] <= context["budget"]["limit_tokens"])
    cache = {"query": "Explain how semantic cache helps agent memory.", "cache_id": "pypi_" + uuid.uuid4().hex}
    cold, warm = call("/api/compare/semantic-cache", cache), call("/api/compare/semantic-cache", cache)
    check("Oracle semantic cache reports a real miss then hit without regeneration", not cold["with"]["cache_hit"] and warm["with"]["cache_hit"] and warm["with"]["metrics"]["provider_calls"] == 0)
    before = call("/api/state")["conversation"]
    compact = call("/api/compare/compaction", {"query": "What travel dates, budget and hotel constraints did I request?"})
    check("Paired compaction uses real summaries and preserves original history", compact["archive"]["lossless_verified"] and not compact["active_history_modified"] and call("/api/state")["conversation"] == before and compact["summary_metrics"]["provider_calls"] > 0)
    job = call("/api/tokenomics", {
        "mode": "custom", "turns": 1, "agents": ["custom", "memorizz", "naive", "decisions"],
        "prompts": ["Use a live web search to check one airline's current cabin baggage rules for my active trip. Cite the airline source and answer in three short sentences."],
        "context_limit": 24000, "options": {},
    })
    while job["status"] == "running":
        time.sleep(2)
        job = call("/api/tokenomics/" + job["id"])
    check("All four agents complete actual live travel search with the PyPI package", job["status"] == "completed" and len(job["rows"]) == 4 and all(row["status"] == "success" and row["provider_calls"] > 0 and row["tool_calls"] > 0 for row in job["rows"]))
    check("Native MemAgent actually invokes the published Anthropic framework adapter", any(call["purpose"] == "Memorizz framework" for row in job["rows"] if row["agent"] == "memorizz" for call in row["calls"]))
finally:
    restored = call("/api/users/select", {"user_id": original["scope"]["user"]})
    check("Full app validation preserves the original traveler", all(restored[key] == original[key] for key in ["scope", "session", "preferences", "conversation"]))

(ROOT / "pypi_travel_validation.json").write_text(json.dumps({"status": "passed", "checks": checks, "memorizz_version": call("/api/status")["memorizz_version"], "validation_user": fresh["scope"]["user"], "job_id": job["id"], "rows": job["rows"], "provider_cache_read_tokens": sum(row["cache_read_tokens"] for row in result["calls"])}, indent=2))
