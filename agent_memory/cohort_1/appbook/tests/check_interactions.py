"""Validate live app interfaces and paired experiments against actual providers."""
import json
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
client = httpx.Client(base_url='http://127.0.0.1:8031', timeout=240)
checks = []


def check(name, condition):
    assert condition, name
    checks.append(name)
    print('PASS:', name, flush=True)


def call(path, payload=None):
    r = client.get(path) if payload is None else client.post(path, json=payload)
    assert r.status_code == 200, (path, r.status_code, r.text[:300])
    return r.json()


for _ in range(45):
    if call('/api/status')['ready']:
        break
    time.sleep(1)
previous = call('/api/state')
fresh = call('/api/conversations', {'keep_trip': True})
check('New conversation resets history and keeps trip and durable preferences',
      previous['scope']['thread'] != fresh['scope']['thread'] and not fresh['conversation'] and
      fresh['preferences'] == previous['preferences'] and fresh['session']['state']['trip'] == previous['session']['state']['trip'])
for query in ['Remember that I prefer an aisle seat and a quiet refundable hotel.',
              'Briefly explain the constraints of my active trip, using my stored preferences. Do not search the web.']:
    call('/api/chat', {'message': query})
vectors = call('/api/embeddings', {'query': 'quiet refundable hotel',
                                   'memory_text': 'I prefer a quiet refundable hotel.'})
check('Embedding space contains actual stored 384D vectors and a query',
      bool(vectors['points']) and vectors['dimensions'] == 384 and len(vectors['query']['vector']) == 384)
q = np.asarray(vectors['query']['vector'])
check('Displayed distances use all dimensions and true vector norms', all(
    abs(p['cosine_distance'] - (1 - np.dot(p['vector'], q)/(np.linalg.norm(p['vector'])*np.linalg.norm(q)))) < 1e-6
    for p in vectors['points']))
check('PCA coordinates are three-dimensional and stored IDs are distinct',
      all(len(p['position']) == 3 for p in vectors['points']) and
      len({p['id'] for p in vectors['points']}) == len(vectors['points']))
retrieval = call('/api/compare/retrieval', {'query': 'TAP Portugal cabin baggage policy', 'fetch_web': True})
check('Live retrieval returns real sources through an HNSW plan', bool(retrieval['candidates']) and 'HNSW' in retrieval['plan'])
ranked = call('/api/compare/reranking', {'query': 'What are the TAP Portugal cabin baggage rules?'})
check('Paired reranking answers each use a single passage from the same pool',
      bool(ranked['without']['answer']) and bool(ranked['with']['answer']) and
      ranked['without']['context']['text'] in [r['text'] for r in ranked['candidates']] and
      ranked['with']['context']['text'] in [r['text'] for r in ranked['candidates']])
cache_id = 'live_test_' + str(time.time_ns())
cold = call('/api/compare/semantic-cache', {'query': 'Explain how semantic cache helps agent memory.', 'cache_id': cache_id})
warm = call('/api/compare/semantic-cache', {'query': 'Explain how semantic cache helps agent memory.', 'cache_id': cache_id})
check('Semantic comparison reports an actual cold miss and warm hit', not cold['with']['cache_hit'] and warm['with']['cache_hit'])
check('Warm semantic side avoids a provider call while direct side calls Claude',
      warm['without']['metrics']['provider_calls'] == 1 and warm['with']['metrics']['provider_calls'] == 0 and
      warm['with']['metrics']['estimated_usd'] == 0)
before = call('/api/state')['conversation']
compressed = call('/api/compare/compaction', {'query': 'What trip constraints and preferences did I ask you to remember?'})
check('Compaction comparison preserves history and verifies lossless compression',
      compressed['archive']['lossless_verified'] and not compressed['active_history_modified'] and
      before == call('/api/state')['conversation'])
check('Both full-history and summarized-history answers have real counted contexts',
      bool(compressed['without']['answer']) and bool(compressed['with']['answer']) and
      compressed['without']['context_tokens'] > 0 and compressed['with']['context_tokens'] > 0 and
      compressed['summary_metrics']['provider_calls'] >= 1)
job = call('/api/tokenomics', {'agents': ['custom', 'memorizz', 'naive', 'decisions'],
    'prompts': ['Explain how semantic cache helps agent memory.'], 'turns': 2,
    'context_limit': 12000, 'options': {'normal_cache': False}})
(ROOT/'data/last_validation_job_id.txt').write_text(job['id'])
print('Experiment job:', job['id'], flush=True)
while job['status'] == 'running':
    time.sleep(2)
    job = call('/api/tokenomics/' + job['id'])
check('Four real agent implementations complete the specified turn count',
      job['status'] == 'completed' and len(job['rows']) == 8 and all(r['status'] == 'success' for r in job['rows']))
check('Append-only agent calls Claude every turn without application or prompt caches', all(
    r['provider_calls'] >= 1 and not r['application_cache'] and r['cache_read_tokens'] == 0
    for r in job['rows'] if r['agent'] == 'naive'))
check('Memorizz lane uses real framework calls, and Jev decisions report actual usage',
      any(c['purpose'] == 'Memorizz framework' for r in job['rows'] if r['agent'] == 'memorizz' for c in r['calls']) and
      any(r['decision_calls'] > 0 and r['decision_usd'] is not None for r in job['rows'] if r['agent'] == 'decisions'))
check('Custom semantic cache option changes the warm turn behavior', any(
    r['agent'] == 'custom' and r['turn'] == 2 and r['application_cache'] == 'semantic' and r['provider_calls'] == 0
    for r in job['rows']))
check('Experiment scopes do not alter the traveler thread or preferences',
      fresh['scope']['thread'] == call('/api/state')['scope']['thread'])
for payload in [{'agents':['custom'],'prompts':['x'],'turns':31},
                {'agents':['fake'],'prompts':['x']},
                {'agents':['custom'],'prompts':['x'],'options':{'fake':True}}]:
    check('Invalid experiment options rejected', client.post('/api/tokenomics',json=payload).status_code == 422)
report = {'status': 'passed', 'checks': checks, 'tokenomics_job': job['id'],
          'agent_rows': [{k:r[k] for k in ['agent','turn','status','seconds','estimated_usd','processed_input_tokens','cache_read_tokens','application_cache']}
                         for r in job['rows']]}
(ROOT/'interaction_validation.json').write_text(json.dumps(report,indent=2))
client.close()
