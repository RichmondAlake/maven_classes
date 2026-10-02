"""Check ownership while a benchmark temporarily changes the core's scope."""
import ast
import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def get_job(job, selected="traveler"):
    source = ROOT / "backend/tokenomics.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get")
    env = {
        "copy": copy,
        "JOB_LOCK": threading.RLock(),
        "JOBS": {"job": job},
        "rt": SimpleNamespace(
            DEFAULT_SCOPE=SimpleNamespace(owner="original"),
            ACTIVE_SCOPE=SimpleNamespace(owner=selected),
            # This owner belongs to an executing benchmark lane, not the UI traveler.
            core=SimpleNamespace(SCOPE=SimpleNamespace(owner="benchmark-lane")),
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), env)
    return env["get"]("job")


def test_selected_user_can_poll_while_benchmark_scope_is_active():
    job = {"id": "job", "owner": "traveler", "rows": []}
    result = get_job(job)
    assert result == job
    result["rows"].append("changed")
    assert job["rows"] == []


def test_another_users_job_is_rejected():
    with pytest.raises(ValueError, match="traveler profile"):
        get_job({"id": "job", "owner": "another-traveler"})


def test_lane_scope_does_not_grant_ui_access():
    with pytest.raises(ValueError, match="traveler profile"):
        get_job({"id": "job", "owner": "benchmark-lane"})


def test_legacy_exports_are_available_to_original_traveler_only():
    assert get_job({"id": "job"}, selected="original")["id"] == "job"
    with pytest.raises(ValueError, match="traveler profile"):
        get_job({"id": "job"})


@pytest.mark.parametrize("stored,expected", [("window", "window"), ('"window"', "window"), ("true", True), (True, True)])
def test_entity_adapter_accepts_native_and_bridge_values(stored, expected):
    source = ROOT.parent / "tools/memorizz_components.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_entity_value")
    env = {"json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), env)
    assert env["decode_entity_value"](stored) == expected


def test_first_chat_extraction_failure_does_not_reuse_a_previous_users_trace():
    source = ROOT / "backend/runtime.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "chat")

    def failed_extraction(request):
        raise RuntimeError("Provider temporarily unavailable")

    core = SimpleNamespace(
        CALLS=[], SEARCH_CALLS=[], SCOPE="selected-user",
        LAST_AGENT_RUN={"inspections": [{"context": {"name": "another traveler"}}]},
        read_state=lambda **kwargs: None,
        parse_trip=failed_extraction,
        stable_prefix=lambda: [],
    )
    env = {"require": lambda: core, "ACTIVE_SCOPE": SimpleNamespace(owner="selected-user"),
           "record": lambda *args: None, "user_calls": {}, "user_searches": {}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), env)
    with pytest.raises(RuntimeError, match="Provider temporarily unavailable"):
        env["chat"]("Hello")
    assert env["last_run"]["inspections"] == []
