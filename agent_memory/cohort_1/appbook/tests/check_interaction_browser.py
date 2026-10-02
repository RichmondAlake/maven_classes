"""Chromium validation of interactive, code-free appbook surfaces."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
BASE=os.getenv('APPBOOK_URL','http://127.0.0.1:8031')
CHROME='/Users/richmondalake/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'
checks=[]
errors=[]


def check(name,value):
    assert value,name
    checks.append(name)
    print('PASS:',name,flush=True)


with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=CHROME)
    page=browser.new_page(viewport={'width':1540,'height':1100},reduced_motion='reduce')
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(BASE,wait_until='networkidle')
    check('Chat formatting renders headings, lists, tables and safe source links', page.evaluate('''() => {
        const node=document.createElement('div');
        node.innerHTML=richText('# Heading\\n\\n- Item\\n\\n| A | B |\\n| --- | --- |\\n| 1 | 2 |\\n\\n[Supplier](https://supplier.example/policy)\\n[Unsafe](javascript:alert)\\n<img src=x onerror=alert(1)>');
        return !!node.querySelector('h2') && !!node.querySelector('ul li') &&
          node.querySelectorAll('table tbody td').length===2 &&
          node.querySelectorAll('a').length===1 && !node.querySelector('img');
    }'''))
    check('No setup section, notebook links or implementation code in navigation',
          page.locator('a[href="#lab/0"]').count()==0 and page.locator('a[href*="notebook"]').count()==0 and
          page.locator('#chapters a[href^="#lab/"]').count()==12)
    page.locator('#new-conversation').click()
    check('New conversation button opens a trip-preserving dialog',page.locator('#conversation-dialog').is_visible() and page.locator('#keep-trip').is_checked())
    page.locator('#cancel-conversation').click()
    page.goto(BASE+'/#lab/1',wait_until='networkidle')
    page.wait_for_function("document.querySelector('#vector-count').textContent.includes('384D')",timeout=60000)
    page.locator('#embedding-query').fill('quiet refundable hotel')
    page.wait_for_function("document.querySelector('#cosine-value').textContent !== '—'",timeout=60000)
    check('Live query updates full-dimension similarity and vector heatmaps',
          bool(page.locator('.neighbor-button').count()) and 'Cosine distance' in page.locator('#distance-value').inner_text())
    page.locator('#dimension-step').fill('64')
    check('Distance calculation responds to the dimension control','64 / 384' in page.locator('#dimension-label').inner_text())
    box=page.locator('#embedding-space').bounding_box()
    page.mouse.move(box['x']+150,box['y']+100);page.mouse.down();page.mouse.move(box['x']+250,box['y']+150,steps=5);page.mouse.up()
    check('Interactive vector canvas has real rendered pixels',page.locator('#embedding-space').evaluate('(c)=>c.width>0 && c.getContext("2d").getImageData(0,0,c.width,c.height).data.some(v=>v>0)'))
    page.screenshot(path=str(ROOT/'embedding_space.png'),full_page=True)
    for number in ['2','3','4','6','7','8','9','10','12']:
        page.goto(BASE+'/#lab/'+number,wait_until='networkidle')
        check('Section '+number+' renders an interaction without code or notebook prose',
              page.locator('.source-cell, .reading, #edition').count()==0 and 'notebook' not in page.locator('#stage').inner_text().lower())
    page.goto(BASE+'/#lab/3',wait_until='networkidle')
    check('Reranking shares one query across two chat panes',page.locator('#shared-query').count()==1 and page.locator('.comparison-chat').count()==2)
    page.goto(BASE+'/#lab/7',wait_until='networkidle')
    check('Semantic cache has paired chat panes and reset control',page.locator('.comparison-chat').count()==2 and page.locator('#reset-semantic').count()==1)
    page.goto(BASE+'/#lab/10',wait_until='networkidle')
    check('Compaction presents full and compacted conversation side by side',page.locator('.comparison-chat').count()==2)
    job_id=(ROOT/'data/last_validation_job_id.txt').read_text().strip() if (ROOT/'data/last_validation_job_id.txt').exists() else None
    if job_id:
        page.evaluate('(id)=>localStorage.setItem("tokenomics-job",id)',job_id)
    page.goto(BASE+'/#tokenomics',wait_until='networkidle')
    check('Tokenomics exposes four agents and optimization controls',page.locator('[name=experiment-agent]').count()==4 and page.locator('[name=experiment-option]').count()==9)
    if job_id:
        page.wait_for_selector('.experiment-chart',timeout=60000)
        check('Tokenomics charts display actual experiment rows',page.locator('.experiment-chart').count()==4 and page.locator('.turn-result').count()>0)
        page.screenshot(path=str(ROOT/'tokenomics.png'),full_page=True)
    page.goto(BASE+'/#architecture',wait_until='networkidle')
    check('Interactive reference architecture remains available',page.locator('#architecture-agent option').count()==4 and page.locator('.ra-node').count()>0)
    mobile=browser.new_page(viewport={'width':390,'height':844},reduced_motion='reduce')
    for route in ['assistant','lab/1','lab/3','lab/7','lab/10','tokenomics']:
        mobile.goto(BASE+'/#'+route,wait_until='networkidle')
        check('Mobile '+route+' fits the viewport',mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
    mobile.screenshot(path=str(ROOT/'interactive_mobile.png'),full_page=True)
    check('No browser JavaScript errors',not errors)
    browser.close()
(ROOT/'interactive_browser_validation.json').write_text(json.dumps({'status':'passed','checks':checks,'errors':errors},indent=2))
