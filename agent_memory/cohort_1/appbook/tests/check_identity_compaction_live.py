"""Real LLM identity extraction, native recall and compaction in an isolated thread."""
import getpass,importlib,json,os,uuid
from dataclasses import replace
from pathlib import Path
import requests
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[1]
private=dotenv_values('/Users/richmondalake/Desktop/llmcamp_ai_hub/system_one_models/.env')
for key in ['ANTHROPIC_API_KEY','TYPESAFE_API_KEY']:
    if private.get(key):os.environ[key]=private[key]
os.environ['TAVILY_API_KEY']=getpass.getpass('TAVILY_API_KEY (hidden): ')
for key,value in json.loads((ROOT/'data/oracle_v2.json').read_text()).items():os.environ[key]=value
os.environ.update(NOTEBOOK_USE_ENV_KEYS='1',ANTHROPIC_MODEL='claude-opus-5-5')
initial=requests.get('http://127.0.0.1:8031/api/state',timeout=180).json()
c=importlib.import_module('backend.memorizz_core')
checks=[]

def check(name,condition):
    assert condition,name
    checks.append(name);print('PASS:',name,flush=True)

for name in ['notebook_core','memorizz_core','decision_core']:
    module=importlib.import_module('backend.'+name)
    positive=module.extract_entities('Remember my name is richmond.')
    negative=module.extract_entities('What is my name? Does the hotel manager named Alex remember me?')
    check(name+' extracts only explicitly stated traveler identity',positive['preferences'].get('name','').lower()=='richmond' and 'name' not in negative['preferences'])
    correction=module.extract_entities("For this experiment only, I'd like a window seat instead of an aisle seat. Keep my quiet-room preference.")
    check(name+' stores scoped corrections rather than the superseded value',correction['preferences'].get('seat')=='window')
    if name=='decision_core':
        check('Jev gates explicit names, not identity questions',module.entity_present('Remember my name is richmond.') and not module.entity_present('What is my name?'))

import sys
if '--extraction-only' in sys.argv:
    (ROOT/'entity_profile_revision_validation.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2))
    raise SystemExit(0)

scope=replace(c.SCOPE,thread_id='identity_validation_'+uuid.uuid4().hex[:12])
check('Native profile is durable across a fresh thread',c.load_profile(scope)['name']==initial['preferences']['name'])
entity=c.entity_memory.get_entity_by_name('traveler',memory_id=scope.owner,user_id=scope.owner)
name_attribute=next(a for a in entity['attributes'] if a['name']=='name')
check('Name is stored in native EntityMemory with original provenance',name_attribute['source']==initial['preferences']['name']['source_turn_id'])
c.set_trip(initial['session']['state']['trip'],scope)
first=c.agent_turn('What is my remembered name and current destination? Answer briefly; no new research is needed.',scope=scope)
check('Fresh conversation recalls durable name and active trip','richmond' in first['answer'].lower() and 'lisbon' in first['answer'].lower())
second=c.agent_turn('Remind me of my saved seat and hotel preferences in one sentence.',scope=scope)
from backend import runtime as rt,interactions
rt.core,rt.ready=c,True
c.SCOPE=scope
result=interactions.compact_now()
check('Compaction persists a real summary and linked original turns',result['compacted']==2 and result['summary'] and result['memory_id'] and result['originals_preserved'])
check('Just-in-time expansion returns original conversations',all(t['event_id'] in c.unpack_memory(result['memory_id'],scope=scope)['content'] for t in c.load_blob(result['memory_id'],scope)['source_turns']))
check('Compaction reports measured input tokens before and after',result['before_tokens']>0 and result['after_tokens']>0 and result['metrics']['provider_calls']>0)
final=requests.get('http://127.0.0.1:8031/api/state',timeout=180).json()
check('Identity/compaction checks preserve active traveler conversation',final['scope']==initial['scope'] and final['preferences']==initial['preferences'])
(ROOT/'identity_compaction_live_validation.json').write_text(json.dumps({'status':'passed','checks':checks,'compaction':result,'fresh_conversation_answer':first['answer'],'thread':scope.thread_id},indent=2,default=str))
print('Validation complete.',flush=True)
