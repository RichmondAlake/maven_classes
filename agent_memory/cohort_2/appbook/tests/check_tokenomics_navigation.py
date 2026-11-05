"""Deliver a real experiment-start response after leaving its page."""
import json,time,os
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
BASE=os.getenv('APPBOOK_URL','http://127.0.0.1:8031')
CHROME='/Users/richmondalake/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=CHROME)
    page=browser.new_page(viewport={'width':1540,'height':1100})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE+'/#tokenomics',wait_until='networkidle')
    page.locator('#experiment-mode').select_option('custom')
    page.locator('#experiment-turns').fill('1');page.locator('#experiment-turns').dispatch_event('change')
    page.locator('#experiment-prompts').fill('Explain agent memory in one sentence.')
    for box in page.locator('[name=experiment-agent]').all():box.set_checked(box.get_attribute('value')=='custom')
    observed={}
    def late_start(route):
        response=route.fetch()
        observed.update(response.json())
        page.evaluate("location.hash='#architecture'")
        page.wait_for_selector('#architecture-agent')
        route.fulfill(response=response)
    page.route('**/api/tokenomics',late_start)
    page.locator('#run-experiment').click()
    page.wait_for_selector('#architecture-agent')
    page.wait_for_function('(id)=>localStorage.getItem("tokenomics-job")===id',arg=observed['id'])
    requests.post(BASE+'/api/tokenomics/'+observed['id']+'/cancel',json={},timeout=30)
    page.wait_for_timeout(1500)
    assert not errors,errors
    assert page.locator('#architecture-agent').is_visible()
    job_id=observed['id']
    print('PASS: Late start response preserves the job without touching the departed page.',flush=True)
    print('PASS: No null-innerHTML errors after navigation.',flush=True)
    browser.close()
while True:
    job=requests.get(BASE+'/api/tokenomics/'+job_id,timeout=60).json()
    if job['status']!='running':break
    time.sleep(2)
(ROOT/'tokenomics_navigation_validation.json').write_text(json.dumps({'status':'passed','job_id':job_id,'errors':errors,'measured_rows':len(job['rows'])},indent=2))
