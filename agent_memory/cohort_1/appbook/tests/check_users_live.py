"""Create empty users, learn actual facts and return to the original traveler."""
import json,os
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
BASE=os.getenv('APPBOOK_URL','http://127.0.0.1:8031')
checks=[]

def check(name,condition):
    assert condition,name
    checks.append(name);print('PASS:',name,flush=True)

def post(path,payload):
    response=requests.post(BASE+path,json=payload,timeout=300)
    if not response.ok:raise RuntimeError(f'{path}: {response.status_code} {response.text[:200]}')
    return response.json()

original=requests.get(BASE+'/api/state',timeout=180).json()
status=requests.get(BASE+'/api/status',timeout=30).json()
check('App uses the published Memorizz package',status['memorizz_version']=='0.13.0')
users=[]
try:
    a=post('/api/users',{'display_name':'Alex · validation'})
    users.append(a['scope']['user'])
    check('New user starts with empty durable and working memories',a['session'] is None and a['preferences']=={} and not any(a[k] for k in ['conversation','workflow','pointers','usage']) and a['scope']['user']!=original['scope']['user'])
    tables=requests.get(BASE+'/api/tables',timeout=90).json()
    check('Memory explorer does not expose another traveler’s rows',all(t['count']==0 for t in tables if t['name']!='AM_STATE_V2'))
    known_job=(ROOT/'data/last_validation_job_id.txt').read_text().strip()
    check('Other traveler’s experiment cannot be opened',requests.get(BASE+'/api/tokenomics/'+known_job,timeout=30).status_code==409)
    remembered=post('/api/chat',{'message':'Remember my name is Alex and I prefer a window seat.'})
    a=remembered['snapshot']
    check('Actual Claude extraction stores new-user identity and preferences',a['preferences']['name']['value'].lower()=='alex' and a['preferences']['seat']['value']=='window')
    check('Saving identity and seat preferences does not invent an active trip',a['session'] is None)
    b=post('/api/users',{'display_name':'Jordan · validation'})
    users.append(b['scope']['user'])
    check('Second user starts empty without Alex or Richmond facts',b['preferences']=={} and b['session'] is None and not b['conversation'])
    returned=post('/api/users/select',{'user_id':users[0]})
    check('Switching back restores profile, trip and last conversation',returned['scope']==a['scope'] and returned['preferences']==a['preferences'] and returned['conversation']==a['conversation'] and returned['session']==a['session'])
    fresh=post('/api/conversations',{'keep_trip':False})
    check('New conversation stays in this user while preserving entity memory',fresh['scope']['user']==users[0] and fresh['scope']['thread']!=a['scope']['thread'] and fresh['preferences']==a['preferences'] and not fresh['conversation'] and fresh['session'] is None)
    recalled=post('/api/chat',{'message':'What is my remembered name and seat preference? Answer briefly.'})
    check('Actual model recalls only the selected user’s facts','alex' in recalled['answer'].lower() and 'window' in recalled['answer'].lower() and 'richmond' not in recalled['answer'].lower() and 'lisbon' not in recalled['answer'].lower())
    denied=requests.post(BASE+'/api/users/select',json={'user_id':'__unregistered_traveler__'},timeout=30)
    check('Unknown profile selection is rejected',denied.status_code==409)
finally:
    restored=post('/api/users/select',{'user_id':original['scope']['user']})
    check('Original traveler retains their trip, profile and conversation',restored['scope']==original['scope'] and restored['preferences']==original['preferences'] and restored['conversation']==original['conversation'] and restored['session']==original['session'])
(ROOT/'users_live_validation.json').write_text(json.dumps({'status':'passed','checks':checks,'users_created':users,'restored_scope':restored['scope'],'package':'memorizz==0.13.0'},indent=2))
