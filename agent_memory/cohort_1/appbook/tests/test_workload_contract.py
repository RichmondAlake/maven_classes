"""Workload sequencing, lane cache invalidation and explicit identity validation."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


def functions(path,names,env=None):
    tree=ast.parse(path.read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace=env or {}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def workload_env():
    path=ROOT/'backend/workloads.py'
    tree=ast.parse(path.read_text())
    scenarios=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCENARIOS' for t in n.targets))
    scope=SimpleNamespace(owner='owner',thread_id='thread')
    return functions(path,{'preset','resolve'},{'SCENARIOS':scenarios,'WORKLOADS':{},'rt':SimpleNamespace(require=lambda:SimpleNamespace(SCOPE=scope))})


@pytest.mark.parametrize('scenario',['research','preferences','context'])
def test_scenarios_evolve_without_implicit_prompt_repetition(scenario):
    sequence=workload_env()['preset'](scenario,30)
    assert len(sequence)==30 and len(set(sequence))==30


def test_cache_scenario_repeats_deliberately():
    sequence=workload_env()['preset']('cache',12)
    assert len(sequence)==12 and sequence[0]==sequence[1]


def test_custom_requests_cannot_silently_cycle():
    with pytest.raises(ValueError,match='every turn'):
        workload_env()['resolve']({'mode':'custom','prompts':['One request'],'turns':4})


@pytest.mark.parametrize('owner,thread,turns',[('foreign','thread',2),('owner','foreign',2),('owner','thread',3)])
def test_workload_cannot_cross_scope_or_changed_turn_count(owner,thread,turns):
    env=workload_env()
    env['WORKLOADS']['id']={'owner':owner,'thread':thread,'prompts':['first','second']}
    with pytest.raises(ValueError):env['resolve']({'workload_id':'id','turns':turns})


def test_profile_corrections_change_cache_identity_but_source_id_alone_does_not():
    profile={'name':{'value':'Richmond','source_turn_id':'first'}}
    c=SimpleNamespace(read_state=lambda **kwargs:{'state':{'trip':{'destination':'Lisbon'}}},load_profile=lambda scope:profile)
    fn=functions(ROOT/'backend/tokenomics.py',{'cache_facts'})['cache_facts']
    before=json.dumps(fn(c,'scope'),sort_keys=True)
    profile['name']['source_turn_id']='second'
    assert json.dumps(fn(c,'scope'),sort_keys=True)==before
    profile['name']['value']='Updated name'
    assert json.dumps(fn(c,'scope'),sort_keys=True)!=before


@pytest.mark.parametrize('name',['', 'A'*121, 'Name\nInjected'])
def test_invalid_identity_names_are_rejected(name):
    fn=functions(ROOT/'backend/notebook_core.py',{'validate_entities'})['validate_entities']
    with pytest.raises(ValueError):fn({'preferences':{'name':name}})
