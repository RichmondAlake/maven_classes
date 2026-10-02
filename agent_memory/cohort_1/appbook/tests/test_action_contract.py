"""Action allowlist and bounded correction tests, without database initialization."""
import ast
import json
from pathlib import Path
import nbformat
import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'implementing_memory_aware_agents.ipynb'


def adapter(responses):
    iterator = iter(responses)
    calls = []
    def model(prompt, **kwargs):
        calls.append(prompt)
        return next(iterator)
    scope = {'json': json, 'llm_json': model, 'stable_prefix': lambda: [],
             'pretty': lambda v: json.dumps(v)}
    for cell in nbformat.read(SOURCE, 4).cells:
        if cell.cell_type != 'code':
            continue
        nodes = [node for node in ast.parse(cell.source).body
                 if isinstance(node, ast.FunctionDef) and node.name in {'validate_action', 'next_action'}
                 or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'ACTION_FIELDS', 'TOOL_PROTOCOL'} for t in node.targets)]
        if nodes:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), 'teaching_action_cells.py', 'exec'), scope)
    return scope, calls


def test_valid_action_does_not_retry():
    scope, calls = adapter([{'action': 'search_travel', 'kind': 'hotel', 'query': 'quiet hotel'}])
    assert scope['next_action']({})['kind'] == 'hotel'
    assert len(calls) == 1


def test_extra_fields_are_rejected_before_correction():
    scope, calls = adapter([{'action': 'compact', 'owner': 'another-user'}, {'action': 'compact'}])
    assert scope['next_action']({}) == {'action': 'compact'}
    assert len(calls) == 2
    assert 'validation' in calls[1].lower()


def test_unsupported_write_never_becomes_a_tool_and_retry_is_bounded():
    scope, calls = adapter([{'action': 'book', 'kind': 'flight'}] * 2)
    with pytest.raises(ValueError):
        scope['next_action']({})
    assert len(calls) == 2


@pytest.mark.parametrize('action', [
    {'action': 'unpack', 'memory_id': 'id', 'offset': -1},
    {'action': 'unpack', 'memory_id': 'id', 'offset': True},
    {'action': 'search_travel', 'kind': 'payment', 'query': 'pay'},
])
def test_invalid_tool_arguments_are_rejected(action):
    scope, _ = adapter([])
    with pytest.raises(ValueError):
        scope['validate_action'](action)
