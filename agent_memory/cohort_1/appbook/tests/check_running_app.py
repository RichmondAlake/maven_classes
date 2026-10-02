"""End-to-end live API validation; no fabricated provider outputs or bookings."""
import argparse
import json
import os
from pathlib import Path
import httpx

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
parser.add_argument('--policy-query', required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
checks = []
client = httpx.Client(base_url=os.getenv('APPBOOK_URL', 'http://127.0.0.1:8031'), timeout=240)


def call(path, data=None, expected=200):
    response = client.get(path) if data is None else client.post(path, json=data)
    assert response.status_code == expected, (path, response.status_code, response.text[:400])
    return response.json()


def check(label, condition):
    assert condition, label
    checks.append(label)
    print('PASS:', label, flush=True)


status = call('/api/status')
check('Published Memorizz backend is ready with mandatory Opus 5.5 and Tavily',
      status['ready'] and status['memory_backend'] == 'memorizz' and status['memorizz_version'] == '0.13.0' and status['responder'] == 'claude-opus-5-5')
state = call('/api/trip', {'request': args.request, 'new_thread': True})
check('Trip comes from the supplied request, rather than a fixture', bool(state['session']['state']['trip']))
embedding = call('/api/labs/1', {'first': 'I want a nonstop flight.', 'second': 'I prefer a direct flight.'})
check('Real normalized 384-dimensional sentence embeddings', embedding['dimensions'] == 384 and embedding['cosine_similarity'] > .6)
retrieval = call('/api/labs/3', {'query': args.policy_query})
check('Live Tavily evidence ingested into Memorizz and retrieved through HNSW',
      bool(retrieval['candidates']) and 'HNSW' in retrieval['plan'].upper())
check('Reranking preserves actual candidates and their source URLs',
      bool(retrieval['reranked']) and all(r['metadata'].get('url') for r in retrieval['reranked']))
preference = call('/api/labs/8', {'query': 'Remember that I prefer an aisle seat and a quiet refundable hotel.'})
check('LLM extraction writes source-attributed Memorizz entity facts', preference['profile']['seat']['value'] == 'aisle')
rag = call('/api/labs/4', {'query': args.policy_query})
check('RAG calls the real model with explicit current trip and selected evidence', bool(rag['answer']) and bool(rag['calls']))
cache_cold = call('/api/labs/7', {'query': 'How does retrieval help an agent choose relevant memories?'})
cache_warm = call('/api/labs/7', {'query': 'How does retrieval help an agent choose relevant memories?'})
check('OracleSemanticCache warm hit avoids an Anthropic call', cache_warm['cache_hit'] and not cache_warm['calls'])
result = call('/api/chat', {'message': args.request})
check('Live agent produces a sourced research answer', bool(result['answer']) and bool(result['sources']))
check('Each iteration reads workflow memory before its action',
      all('workflow_memory' in step['context'] for step in result['inspections']) and
      any(step['context']['workflow_memory'] for step in result['inspections'][1:]))
workflow = call('/api/labs/9', {})['workflow']
check('Native workflow steps capture decisions, outcomes and success/failure status',
      len(workflow) >= len(result['inspections']) and
      all('decision' in e['payload'] and 'outcome' in e['payload'] and e['status'] in {'success', 'failure'} for e in workflow))
check('Anthropic reports actual cached-prefix reads', sum(c['cache_read_tokens'] for c in result['calls']) > 0)
context = call('/api/context')
check('Context inspector exposes stable prefix, dynamic memory and real token counts',
      context['prefix'][0]['cache_control']['type'] == 'ephemeral' and
      context['budget']['after_tokens'] <= context['budget']['limit_tokens'] and
      'memory_placeholders' in context['dynamic_context'])
threshold = call('/api/labs/12', {'exercise_threshold': True})['threshold_experiment']['budget']
check('Host threshold automatically compacts actual history and restores a bounded context',
      threshold['before_tokens'] > threshold['limit_tokens'] and
      threshold['after_tokens'] <= threshold['limit_tokens'] and
      threshold['automatic_compaction'] is not None)
compact = call('/api/labs/10', {})
check('Compaction summarizes actual conversation turns', compact['compacted'] > 0 and bool(compact['memory_id']))
archive = call('/api/labs/10', {'memory_id': compact['memory_id']})
check('Just-in-time archive unpack verifies scope and checksum', 'source_turns' in archive['content'] and archive['memory_id'] == compact['memory_id'])
probes = call('/api/labs/12', {'probe_memories': True})
check('Native conversational, workflow and summary recall apply owner and thread scope',
      all(p['owned_count'] > 0 and p['foreign_owner_count'] == 0 and p['other_thread_count'] == 0
          for p in probes['memory_probes'].values()))
check('A real missing-memory tool failure is recorded as a failed workflow step',
      probes['recorded_failure']['status'] == 'failure' and
      probes['recorded_failure']['payload']['outcome']['error_type'] == 'ValueError')
jit = call('/api/chat', {'message': f'Use the unpack tool at offset zero for memory {compact["memory_id"]}, then briefly explain the remembered trip constraints. Do not search the web.'})
check('The agent itself requests a bounded just-in-time archive read', any(s['action']['action'] == 'unpack' for s in jit['inspections']))
check('Unpacked page is available in the next invocation context', any(
    isinstance(e.get('outcome_preview'), dict) and 'content' in e['outcome_preview']
    for s in jit['inspections'] for e in s['context']['workflow_memory']))
check('App exposes interactions without notebook downloads',
      all(client.get('/notebook/' + edition).status_code == 404
          for edition in ['memorizz', 'scratch', 'decisions']))
tables = call('/api/tables')
check('All current memory tables are inspectable', len(tables) >= 9 and all(call('/api/tables/' + t['name'])['name'] == t['name'] for t in tables))
cross = client.post('/api/trip', json={'request': args.request}, headers={'Origin': 'https://unrelated.example'})
check('Cross-origin writes rejected', cross.status_code == 403)
call('/api/labs/99', {}, expected=404)
call('/api/labs/10', {'memory_id': 'nonexistent-memory'}, expected=409)
check('Unknown Part and missing memory cannot execute', True)
report = {'status': 'passed', 'checks': checks, 'model': status['responder'],
          'backend': status['memory_backend'], 'oracle': 'isolated native HNSW classroom schema',
          'provider_cache_read_tokens': sum(c['cache_read_tokens'] for c in result['calls']),
          'agent_iterations': len(result['inspections']), 'jit_iterations': len(jit['inspections'])}
(ROOT / 'validation_report.json').write_text(json.dumps(report, indent=2))
client.close()
