"""Scope, request-budget and cost boundaries without initializing live providers."""
import ast
import copy
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "backend/tokenomics.py"


def definitions():
    tree = ast.parse(SOURCE.read_text())
    names = {"educational", "totals", "controls", "retrieve_tool_log_entry"}
    selected = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    env = {
        "copy": copy,
        "json": json,
        "hashlib": hashlib,
        "time": time,
        "contextmanager": contextmanager,
        "np": np,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), env)
    return env


def test_missing_search_usage_does_not_become_free():
    result = definitions()["totals"]([], [{"usage": None}], [])
    assert result["estimated_usd"] is None and result["search_usd"] is None
    assert result["provider_calls"] == 0 and result["tool_calls"] == 1


def test_provider_input_counters_are_disjoint_and_costs_include_decisions():
    calls = [
        {
            "estimated_usd": 0.01,
            "input_tokens": 100,
            "cache_read_tokens": 200,
            "cache_write_tokens": 300,
            "output_tokens": 40,
        }
    ]
    result = definitions()["totals"](
        calls, [{"usage": {"credits": 1}}], [{"usage": {"input_tokens": 1000}}]
    )
    assert result["processed_input_tokens"] == 600
    assert result["estimated_usd"] == pytest.approx(0.018042)


@pytest.mark.parametrize(
    "query",
    [
        "Search current flights",
        "Explain semantic cache for my itinerary",
        "Remember that I prefer a quiet hotel",
        "What is the current hotel price?",
    ],
)
def test_traveler_requests_are_ineligible_for_answer_reuse(query):
    assert not definitions()["educational"](query)


def archive_reader(entry):
    env = definitions()
    env["c"] = SimpleNamespace(pretty=lambda value: json.dumps(value))
    env["self"] = SimpleNamespace(
        scope=SimpleNamespace(owner="lane", thread_id="thread"),
        framework=SimpleNamespace(
            memory_manager=SimpleNamespace(retrieve_tool_log=lambda *a, **k: entry)
        ),
    )
    return env["retrieve_tool_log_entry"]


def test_archive_expansion_is_bounded_and_original_is_reconstructable():
    content = "Policy evidence. " * 700
    raw = json.dumps(
        {
            "results": [
                {
                    "url": "https://supplier.example/policy",
                    "title": "Policy",
                    "content": content,
                }
            ]
        }
    )
    reader = archive_reader({"memory_id": "lane", "thread_id": "thread", "result": raw})
    offset, pages = 0, []
    while offset is not None:
        page = reader("id", offset=offset, max_chars=100000)
        assert len(page["content"]) <= 2400
        assert (
            page["complete_original_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
        )
        assert page["url"] == "https://supplier.example/policy"
        pages.append(page["content"])
        offset = page["next_offset"]
    assert "".join(pages) == content


@pytest.mark.parametrize("owner,thread", [("another", "thread"), ("lane", "another")])
def test_archive_expansion_rejects_other_lane_or_thread(owner, thread):
    reader = archive_reader({"memory_id": owner, "thread_id": thread, "result": "{}"})
    with pytest.raises(ValueError):
        reader("id")


def test_disabled_options_and_scope_are_restored_after_failure():
    env = definitions()
    names = [
        "SCOPE",
        "stable_prefix",
        "rerank",
        "embed",
        "web_search",
        "search_travel",
        "workflow_memory",
        "workflow_capsule",
        "bounded_context",
        "CONTEXT_LIMIT",
        "read_state",
        "load_profile",
        "compact_thread",
    ]
    core = SimpleNamespace(**{name: (lambda *a, **k: None) for name in names})
    core.SCOPE, core.CONTEXT_LIMIT = "traveler", 100
    core.stable_prefix = lambda: [
        {"text": "policy", "cache_control": {"type": "ephemeral"}}
    ]
    core.read_state = lambda slot, thread, scope: scope
    original = vars(core).copy()
    lane = SimpleNamespace(
        c=core,
        scope="isolated",
        context_limit=200,
        identity="run:custom",
        options={
            name: False
            for name in [
                "prompt_cache",
                "reranking",
                "embedding_cache",
                "tool_cache",
                "offloading",
                "workflow_recall",
                "compaction",
            ]
        },
    )
    with pytest.raises(RuntimeError, match="deliberate failure"):
        with env["controls"](lane):
            assert core.SCOPE == "isolated" and core.read_state() == "isolated"
            assert "cache_control" not in core.stable_prefix()[0]
            assert core.rerank("q", ["first", "second"], 1) == ["first"]
            with pytest.raises(RuntimeError, match="disabled"):
                core.compact_thread()
            raise RuntimeError("deliberate failure")
    assert vars(core) == original
