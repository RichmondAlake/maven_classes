"use strict";
function renderTokenomics(){
  let jobId=localStorage.getItem('tokenomics-job'),timer=null,closed=false,latest=null,workload=null,previewTicket=0;
  const hidden=new Set();
  $('#stage').innerHTML=heading(null,'Compare the <em>tokenomics.</em>','Run the same evolving conversation through independent agents. Measure actual latency, token usage and API cost.')+`
  <section class="panel"><div class="panel-head"><h2>Experiment controls</h2><span class="pill">Opus 5.5 for all answers</span></div><div class="panel-body"><div class="comparison-grid"><div>
    <label for="experiment-mode">Turn source</label><select id="experiment-mode"><option value="scenario">Choose a scenario</option><option value="synthetic">Generate synthetic turns with Claude</option><option value="custom">Write my own turn sequence</option></select>
    <label for="experiment-scenario">Conversation scenario</label><select id="experiment-scenario">${Object.entries(SCENARIO_INFO).map(([k,v])=>`<option value="${k}">${v[0]}</option>`).join('')}</select><p class="hint" id="scenario-hint"></p>
    <div id="synthetic-focus" hidden><label for="experiment-focus">Generation focus · optional</label><textarea id="experiment-focus" placeholder="For example, introduce a preference correction and a later recall question"></textarea><p class="hint">Claude creates the user messages. Agents still make real calls and produce their own answers. Generated preferences remain inside experiment scopes.</p></div>
    <label for="experiment-turns">Turns per agent</label><input id="experiment-turns" type="number" min="1" max="30" value="4">
    <div class="actions"><button class="button secondary small" id="preview-workload">Preview turn sequence</button></div><div id="workload-status" class="hint"></div>
    <label for="experiment-prompts">Resolved turns · one request per line</label><textarea id="experiment-prompts" rows="7" readonly placeholder="Preview a scenario or generate synthetic turns"></textarea>
    <p class="hint">Every agent receives this exact sequence and the same starting trip/profile in separate memory scopes. Only the cache scenario deliberately repeats requests.</p>
    <label for="experiment-limit">Memory-agent context budget</label><input id="experiment-limit" type="number" min="3000" max="100000" step="1000" value="24000">
  </div><div><span class="eyebrow">AGENTS</span><div class="option-grid">${Object.entries(AGENT_INFO).map(([k,[label,hint]])=>`<label class="option-hint"><span><input type="checkbox" name="experiment-agent" value="${k}" checked ${k==='decisions'&&!status.decision_available?'disabled':''}> ${label}</span><small>${hint}</small></label>`).join('')}</div>
    <span class="eyebrow">CUSTOM AND DECISION AGENT OPTIONS</span><div class="option-grid">${Object.entries(OPTION_INFO).map(([k,[label,hint]])=>`<label class="option-hint"><span><input type="checkbox" name="experiment-option" value="${k}" ${selectedExperimentOptions()[k]?'checked':''}> ${label}</span><small>${hint}</small></label>`).join('')}</div>
    <p class="hint">Memorizz uses a fixed native preset with answer, embedding and tool caches, prompt caching, scoped recall, compaction and offloading; cross-encoder reranking is off. Append-only retains all messages and tool results.</p><a href="#architecture" class="button secondary small">Explore each agent architecture ↗</a>
  </div></div><div class="actions"><button class="button" id="run-experiment">Run live experiment ↗</button><button class="button secondary" id="stop-experiment" disabled>Stop after current turn</button><button class="button secondary" id="export-experiment" disabled>Export observed data</button></div></div></section><div class="metrics-strip" id="experiment-progress"></div><div id="experiment-results"></div>`;
  function controls(){return Object.fromEntries([...document.querySelectorAll('[name=experiment-option]')].map(x=>[x.value,x.checked]));}
  document.querySelectorAll('[name=experiment-option]').forEach(x=>x.onchange=()=>localStorage.setItem('tokenomics-options',JSON.stringify(controls())));
  function invalidate(){
    workload=null;previewTicket++;const mode=$('#experiment-mode').value;
    $('#synthetic-focus').hidden=mode!=='synthetic';$('#experiment-scenario').disabled=mode==='custom';$('#preview-workload').hidden=mode==='custom';
    $('#preview-workload').textContent=mode==='synthetic'?'Generate and preview turns ↗':'Preview turn sequence';
    $('#experiment-prompts').readOnly=mode!=='custom';$('#experiment-prompts').value='';$('#workload-status').textContent='';
    $('#scenario-hint').textContent=mode==='custom'?'Supply exactly one request per requested turn; no automatic repetition.':SCENARIO_INFO[$('#experiment-scenario').value][1];
  }
  ['experiment-mode','experiment-scenario','experiment-turns','experiment-focus'].forEach(id=>$('#'+id).onchange=invalidate);invalidate();
  async function prepare(){
    const ticket=++previewTicket;
    const payload={mode:$('#experiment-mode').value,scenario:$('#experiment-scenario').value,turns:Number($('#experiment-turns').value),focus:$('#experiment-focus').value.trim()};
    const result=await api('/api/workloads',payload);if(closed||ticket!==previewTicket)return null;
    workload=result;$('#experiment-prompts').value=result.prompts.join('\n');
    const g=result.generation;
    $('#workload-status').textContent=g?`Generated ${result.prompts.length} turns in ${g.seconds.toFixed(2)} s · shared API setup ${g.estimated_usd===null?'Unknown':'$'+g.estimated_usd.toFixed(6)} · charged once, separately from agent comparisons.`:`${result.prompts.length} scenario turns ready · no generation API charge.`;
    return result;
  }
  $('#preview-workload').onclick=()=>busy($('#preview-workload'),prepare);
  function show(job){
    if(closed||!$('#experiment-results'))return;
    latest=job;jobId=job.id;localStorage.setItem('tokenomics-job',jobId);
    const running=job.status==='running';$('#stop-experiment').disabled=!running;$('#run-experiment').disabled=running;$('#run-experiment').dataset.persistentDisabled=String(running);$('#export-experiment').disabled=!job.rows.length;
    $('#experiment-progress').innerHTML=`<span class="pill ${job.status==='failed'?'warn':'good'}">${escapeHTML(job.status)} · ${job.rows.length}/${job.turns*job.agents.length} turns</span>${job.current?`<span class="pill">${AGENT_INFO[job.current.agent][0]} · turn ${job.current.turn}</span>`:''}${job.error?`<p class="error">${escapeHTML(job.error)}</p>`:''}`;
    const totals=job.agents.map(agent=>{
      const rows=job.rows.filter(r=>r.agent===agent),setup=job.setup[agent],known=rows.every(r=>r.estimated_usd!==null)&&setup?.estimated_usd!==null;
      return{agent:AGENT_INFO[agent][0],turns:rows.length,failures:rows.filter(r=>r.status==='failure').length,total_seconds:rows.reduce((n,r)=>n+r.seconds,0).toFixed(2),estimated_usd:known?'$'+(rows.reduce((n,r)=>n+r.estimated_usd,0)+(setup?.estimated_usd||0)).toFixed(6):'Unknown',processed_input_tokens:rows.reduce((n,r)=>n+r.processed_input_tokens,0),output_tokens:rows.reduce((n,r)=>n+r.output_tokens,0),cache_read_tokens:rows.reduce((n,r)=>n+r.cache_read_tokens,0),setup_seconds:setup?.seconds.toFixed(2)||'—'};
    });
    const visible=job.agents.filter(a=>!hidden.has(a)), generation=job.workload?.generation;
    $('#experiment-results').innerHTML=`<div class="legend agent-legend">${job.agents.map(a=>`<button class="legend-toggle agent-${a} ${hidden.has(a)?'series-hidden':''}" data-agent-toggle="${a}" aria-pressed="${!hidden.has(a)}" title="${hidden.has(a)?'Show':'Hide'} ${AGENT_INFO[a][0]} in every chart">● ${AGENT_INFO[a][0]}</button>`).join('')}</div><p class="hint">Click an agent name to show or hide its charts. Measurements remain in the totals and export.</p>
      <div class="chart-grid"><section class="panel"><div class="panel-head"><h2>Latency per turn · seconds</h2></div>${chartSVG(job.rows,visible,'seconds','Measured latency per turn')}</section><section class="panel"><div class="panel-head"><h2>Cumulative API cost · includes lane setup</h2></div>${chartSVG(job.rows,visible,'estimated_usd','Cumulative estimated API cost',true,job.setup)}</section><section class="panel"><div class="panel-head"><h2>Input tokens processed per turn</h2></div>${chartSVG(job.rows,visible,'processed_input_tokens','Processed input tokens')}</section><section class="panel"><div class="panel-head"><h2>Provider cache reads per turn</h2></div>${chartSVG(job.rows,visible,'cache_read_tokens','Provider cache tokens reused')}</section></div>
      <section class="panel"><div class="panel-head"><h2>Observed totals</h2></div><div class="panel-body">${table(totals)}<p class="hint">${escapeHTML(job.cost_scope)}. Lane startup is included in cumulative cost; turn latency excludes startup. Unknown costs remain unknown.</p>${generation?`<div class="hint">Synthetic turn generation: ${generation.seconds.toFixed(2)} s · ${generation.estimated_usd===null?'Unknown cost':'$'+generation.estimated_usd.toFixed(6)} shared setup. This single charge is excluded from each lane’s comparison.${usageTable(generation.calls)}</div>`:''}</div></section>
      ${job.workload?`<section class="panel"><div class="panel-head"><h2>Shared workload · ${escapeHTML(job.workload.mode)} ${job.workload.scenario?' / '+escapeHTML(job.workload.scenario):''}</h2></div><div class="panel-body"><ol>${job.prompts.map(p=>`<li>${escapeHTML(p)}</li>`).join('')}</ol></div></section>`:''}
      <section class="panel"><div class="panel-head"><h2>Turn outcomes</h2></div><div class="panel-body">${job.rows.map(r=>`<details class="turn-result"><summary><span class="agent-${r.agent}">●</span> ${AGENT_INFO[r.agent][0]} · turn ${r.turn} · ${r.seconds.toFixed(2)} s · ${r.estimated_usd===null?'Unknown cost':'$'+r.estimated_usd.toFixed(6)} · ${r.application_cache?r.application_cache+' hit':r.status}</summary><p>${escapeHTML(r.query)}</p>${r.error?`<p class="error">${escapeHTML(r.error)}</p>`:richText(r.answer)}<div class="pref-list">${Object.entries(r.cache_hits).map(([k,v])=>`<span class="pill">${k}: ${v} hits</span>`).join('')}</div>${usageTable(r.calls)}</details>`).join('')}</div></section>`;
    document.querySelectorAll('[data-agent-toggle]').forEach(button=>button.onclick=()=>{const a=button.dataset.agentToggle;hidden.has(a)?hidden.delete(a):hidden.add(a);show(latest);});
    $('#export-experiment').onclick=()=>{const blob=new Blob([JSON.stringify(job,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='tokenomics-'+job.id+'.json';a.click();URL.revokeObjectURL(url);};
  }
  async function poll(){if(closed||!jobId)return;try{const job=await api('/api/tokenomics/'+jobId);if(closed)return;show(job);if(job.status==='running')timer=setTimeout(poll,1200);}catch(e){if(!closed)toast(e.message);}}
  $('#run-experiment').onclick=()=>busy($('#run-experiment'),async()=>{
    const mode=$('#experiment-mode').value,turns=Number($('#experiment-turns').value);
    const agents=[...document.querySelectorAll('[name=experiment-agent]:checked:not(:disabled)')].map(x=>x.value);if(!agents.length)throw new Error('Choose at least one agent.');
    const options=controls(),context_limit=Number($('#experiment-limit').value),scenario=$('#experiment-scenario').value;
    const prompts=$('#experiment-prompts').value.split('\n').map(s=>s.trim()).filter(Boolean);
    if(mode==='custom'&&prompts.length!==turns)throw new Error('Supply exactly one custom request per turn.');
    if(mode==='synthetic'&&!workload)throw new Error('Generate and review the synthetic sequence first.');
    if(mode==='scenario'&&!workload)await prepare();if(closed)return;
    const job=await api('/api/tokenomics',{mode,scenario,prompts,agents,turns,context_limit,options,workload_id:mode==='custom'?null:workload?.id});
    // Preserve the job so a late response can be resumed, without touching a departed view.
    jobId=job.id;localStorage.setItem('tokenomics-job',jobId);if(closed)return;show(job);await poll();
  });
  $('#stop-experiment').onclick=async()=>{if(jobId){try{await api('/api/tokenomics/'+jobId+'/cancel',{});if(!closed)toast('Stopping after the current turn.');}catch(e){if(!closed)toast(e.message);}}};
  viewCleanup=()=>{closed=true;previewTicket++;clearTimeout(timer);};if(jobId)poll();
}
