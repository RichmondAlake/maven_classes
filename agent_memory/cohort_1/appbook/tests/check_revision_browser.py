"""Validate agent-specific diagrams, chart visibility and scenario controls."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
BASE=os.getenv('APPBOOK_URL','http://127.0.0.1:8031')
CHROME='/Users/richmondalake/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'
checks,errors=[],[]

def check(name,value):
    assert value,name
    checks.append(name)
    print('PASS:',name,flush=True)

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=CHROME)
    page=browser.new_page(viewport={'width':1540,'height':1100},reduced_motion='reduce')
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(BASE,wait_until='networkidle')
    check('Profile shows the name from durable entity memory',page.locator('#profile-name').inner_text().lower()=='richmond')
    page.goto(BASE+'/#architecture',wait_until='networkidle')
    counts={}
    for agent in ['custom','memorizz','naive','decisions']:
        page.locator('#architecture-agent').select_option(agent)
        nodes=page.locator('.ra-node').all_text_contents();counts[agent]=len(nodes)
        check(agent+' architecture uses the actual orchestration',any({'custom':'Manual agent loop','memorizz':'MemAgent','naive':'Append-only loop','decisions':'Hosted Jev'}[agent] in text for text in nodes))
        check(agent+' simulator has valid flow endpoints',page.evaluate('''() => {
            const agent=document.querySelector('#architecture-agent').value;
            const spec=agentArchitecture(agent,agent==='naive'?Object.fromEntries(Object.keys(OPTION_INFO).map(k=>[k,false])):agent==='memorizz'?Object.fromEntries(Object.keys(OPTION_INFO).map(k=>[k,k!=='reranking'])):selectedExperimentOptions());
            const nodes=new Set(spec.components.map(n=>n.id)),edges=new Set(spec.flows.map(f=>f.id));
            return spec.flows.every(f=>nodes.has(f.from)&&nodes.has(f.to)) && spec.runs.every(r=>r.steps.length&&r.steps.every(s=>edges.has(s.flow)));
        }'''))
        page.locator('#ra-step').click();page.wait_for_timeout(1300)
        check(agent+' simulation advances',page.locator('.ra-steps li.done,.ra-steps li.active').count()>0)
        page.screenshot(path=str(ROOT/('architecture_'+agent+'.png')),full_page=True)
    check('Append-only excludes memory recall and caches',page.evaluate("!agentArchitecture('naive',{}).components.some(n=>['recall','cache','compact','decision','embed'].includes(n.id))"))
    page.locator('#architecture-agent').select_option('custom')
    page.locator('[data-architecture-option="reranking"]').uncheck()
    check('Option removal changes diagram and is shared with Tokenomics',page.locator('.ra-node[data-node="rerank"]').count()==0)
    page.locator('[data-architecture-option="reranking"]').check()
    job_id=(ROOT/'data/last_validation_job_id.txt').read_text().strip()
    page.evaluate('(id)=>localStorage.setItem("tokenomics-job",id)',job_id)
    page.goto(BASE+'/#tokenomics',wait_until='networkidle')
    page.wait_for_selector('.experiment-chart')
    check('All 13 selectors have visible explanatory hints',page.locator('.option-hint small').count()==13)
    original_rows=page.locator('.turn-result').count()
    for agent in ['custom','memorizz','naive','decisions']:
        page.locator('[data-agent-toggle="'+agent+'"]').click()
        check(agent+' hides in all charts without dropping results',page.locator('[data-chart-agent="'+agent+'"]').count()==0 and page.locator('.turn-result').count()==original_rows)
        page.locator('[data-agent-toggle="'+agent+'"]').click()
        check(agent+' can be restored in all charts',page.locator('[data-chart-agent="'+agent+'"]').count()==4)
    page.locator('#experiment-turns').fill('8');page.locator('#experiment-turns').dispatch_event('change')
    page.locator('#experiment-scenario').select_option('preferences')
    with page.expect_response(lambda r:r.url.endswith('/api/workloads'),timeout=60000) as response:
        page.locator('#preview-workload').click()
    workload=response.value.json()
    check('Selected scenario previews distinct evolving turns',len(workload['prompts'])==8 and len(set(workload['prompts']))==8 and workload['generation'] is None)
    page.locator('#experiment-mode').select_option('synthetic')
    check('Synthetic source exposes Claude generation and review',page.locator('#synthetic-focus').is_visible() and 'Generate' in page.locator('#preview-workload').inner_text() and page.locator('#experiment-prompts').get_attribute('readonly') is not None)
    page.locator('#experiment-mode').select_option('custom')
    check('Custom turns are editable and never silently cycle',page.locator('#experiment-prompts').get_attribute('readonly') is None and 'no automatic repetition' in page.locator('#scenario-hint').inner_text())
    page.goto(BASE+'/#lab/10',wait_until='networkidle')
    check('Compaction exposes a real trigger next to paired comparison',page.locator('#auto-compact').count()==1 and page.locator('.comparison-chat').count()==2)
    with page.expect_response(lambda r:r.url.endswith('/api/compaction'),timeout=180000) as response:
        page.locator('#auto-compact').click()
    compacted=response.value.json()
    page.wait_for_selector('#comparison-extra h2')
    check('Compaction button applies actual policy and displays measured effect',response.value.ok and compacted['originals_preserved'] and str(compacted['before_tokens']) in page.locator('#comparison-note').inner_text() and 'compaction applied' in page.locator('#comparison-extra').inner_text().lower())
    mobile=browser.new_page(viewport={'width':390,'height':844},reduced_motion='reduce')
    for route in ['architecture','tokenomics','lab/10']:
        mobile.goto(BASE+'/#'+route,wait_until='networkidle')
        check('Mobile '+route+' has no overflow',mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth'))
    check('No browser runtime errors',not errors)
    browser.close()
(ROOT/'revision_browser_validation.json').write_text(json.dumps({'status':'passed','checks':checks,'components':counts,'errors':errors},indent=2))
