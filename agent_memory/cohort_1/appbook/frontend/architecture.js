"use strict";
// Each specification follows the corresponding Lane adapter in backend/tokenomics.py.
function agentArchitecture(agent, options) {
  const native = agent === 'memorizz', naive = agent === 'naive', decisions = agent === 'decisions';
  const tiers = [
    {id:'input',title:'Shared experiment inputs',note:'The same resolved turns and starting trip for every agent'},
    {id:'control',title:native?'Memorizz orchestration':naive?'Append-only orchestration':'Custom orchestration',note:native?'Native MemAgent, ContextPolicy and RetrievalPolicy':naive?'In-process messages grow with every turn':'Explicit context construction at every iteration'},
    {id:'models',title:'Actual model calls',note:decisions?'Claude generates; hosted Jev makes typed decisions':'Raw Anthropic generation; local models where used'},
    {id:'tools',title:'Available tools',note:naive?'Live search and full results':'Scoped recall and current web evidence'},
    {id:'memory',title:naive?'In-process transcript':'Oracle AI Database',note:naive?'No persistent recall on the answer path':native?'Native Memorizz memory units and host state':'Versioned state, scoped vectors, events and archives'},
    {id:'observe',title:'Tokenomics measurements',note:'Measured time and provider-reported usage; isolated experiment scopes'},
  ];
  const components=[], flows=[];
  function node(id,tier,kind,name,tech,note){components.push({id,tier,kind,name,tech,note,col:components.filter(c=>c.tier===tier).length});}
  function edge(id,from,to,label,kind,note){if(components.some(c=>c.id===from)&&components.some(c=>c.id===to))flows.push({id,from,to,label,kind,note});}
  node('turns','input','person','Resolved turn sequence','Scenario / generated / custom','Every lane receives exactly the same ordered turns. Generated turns are inputs, not simulated answers.');
  node('starting','input','channel','Starting trip and profile','Snapshot of your active trip','Copy the actual current trip and remembered profile into isolated experiment state. Each lane has an independent history and caches.');
  node('scope','control','control','Experiment scope','Job + agent + thread','The host fixes owner and thread; one lane cannot retrieve another lane’s memories.');
  node('context','control','control',native?'Native context manager':naive?'Append messages':'Context builder',native?'Bounded native memory':naive?'Full messages + tool results':'Stable prefix + fresh memories',native?'MemAgent selects native conversation, knowledge, workflow, entity and summary units with its retrieval policy.':naive?'Append every user request, assistant response and complete tool result. Keep the initial trip and profile as input.':'Read current trip, profile, recent turns, workflow capsules and memory placeholders before each step.');
  node('loop','control','control',native?'MemAgent':naive?'Append-only loop':'Manual agent loop',native?'Native tool-call protocol':'Up to 8 decision steps',native?'The published Memorizz package drives model calls and tools. It is not a wrapper around the custom loop.':'Parse allowed actions, execute tools and produce a response. Failed calls still contribute measured usage.');
  if(!naive&&(options.normal_cache||options.semantic_cache))node('cache','control','control','Stable answer reuse',options.normal_cache&&options.semantic_cache?'Exact + semantic cache':options.normal_cache?'Exact answer cache':'OracleSemanticCache','Only stable educational questions qualify. Namespace includes model, settings, current trip/profile and expiry. Live travel answers are regenerated.');
  node('claude','models','model','Claude Opus 5.5',options.prompt_cache?'Anthropic · prompt cache':'Raw Anthropic client',native?'Native Memorizz Anthropic adapter makes actual SDK calls. Usage is recorded from each response.':naive?'Raw Claude generates the next action or answer from the complete transcript.':'Raw Claude generates answers, extracts explicit identity/preferences and drafts summaries where available.');
  if(!naive)node('embed','models','model','MiniLM encoder',options.embedding_cache?'384D · cached vectors':'384D local vectors','Semantic addresses for Oracle HNSW recall. Embedding reuse saves local computation, not paid embedding API tokens.');
  if(!naive&&options.reranking&&!decisions)node('rerank','models','model','Cross-encoder','Query / passage relevance','Rerank the retrieved candidate pool before selecting answer context.');
  if(decisions)node('decision','models','decide','Hosted Jev','Typed decisions','Entity-presence gate, toolbox/skillbox routing, summary quality gate and, when enabled, passage ranking. This lane actually uses Jev; CLEF is an alternative in the separate implementation, not a model silently used by this experiment.');
  node('search','tools','external','Tavily live search',options.tool_cache?'60-second request cache':'Live evidence + URLs','Research flights, hotels, transport and policies. Complete results are retained; the app has no supplier booking API.');
  if(!naive)node('recall','tools','tool','Oracle HNSW recall',native?'MemoryManager + provider':'langchain_oracledb OracleVS',native?'Native Oracle approximate vector search, thread-scoped conversation/workflow and memory-scoped supplier passages.':'Approximate HNSW retrieval over scoped OracleVS passages; exact nearest-neighbor search is only a comparison baseline.');
  if(!naive&&options.offloading)node('unpack','tools','tool','Offload and unpack',native?'Native logs · bounded pages':'Archive ID + bounded pages','Store large tool results in Oracle and provide a small reference. Read selected original content just in time.');
  if(!naive&&options.compaction)node('compact','tools','tool','Context compaction',native?'Native policy · 60% budget':'Agent tool + host threshold',native?'Native context policy compacts history and retains four recent messages. Tool-result policy offloads above 1,800 characters.':'Use a real LLM summary when requested or when the context threshold is reached. Preserve originals and expose a summary pointer.');
  if(naive)node('transcript','memory','store','Complete transcript','In-process append-only list','All messages and tool payloads remain in the next prompt. Lane setup is persisted for experiment isolation, but answers do not recall Oracle memories.');
  else {
    node('state','memory','store','Trip and entity profile',native?'State bridge + EntityMemory':'Versioned host state','Current constraints and explicit identity/preference facts, including name and source-turn provenance.');
    node('vectors','memory','store','Knowledge vectors',native?'KnowledgeBase + VECTOR':'OracleVS + HNSW graph','Supplier passages retain source URLs and retrieval timestamps.');
    node('history','memory','store','Conversation and workflow',native?'Native memory units':'Conversation + workflow events',options.workflow_recall?'Read previous decisions and outcomes plus bounded recent turns at the next invocation.':'Persist execution outcomes; recall prior-run workflow is disabled for the custom lane. Current-run tool results remain available.');
    if(options.offloading||options.compaction)node('archive','memory','store','Originals and summaries',native?'Native summaries / tool logs':'Compressed BLOB + source links','Summary and tool references lead back to scoped originals. A summary changes context; lossless compression reduces storage.');
  }
  node('usage','observe','observe','Provider usage and cost','Tokens · reads · latency','Record actual Anthropic calls, Tavily credits and Jev tokens. Include lane setup in cumulative cost; workload generation is one shared setup charge.');
  node('charts','observe','observe','Four-agent comparison','Same ordered input turns','Compare observed latency, API cost, processed input and cache reads. Legend visibility changes the charts, never the measurements.');
  edge('ask','turns','scope','next turn','request','Resolve the next shared turn in the lane’s own scope.');
  edge('seed','starting','context','same initial facts','data','Each lane starts with the same actual trip and profile snapshot.');
  edge('load','scope','context','build context','control','Construct this agent’s context for the invocation.');
  edge('begin','context','loop','selected context','data','Use the selected history and facts with this turn.');
  edge('reason','loop','claude','generate / act','request',options.prompt_cache?'Claude receives the stable cached prefix before changing memory data.':'Claude receives the request without provider prompt-cache annotations.');
  edge('route','loop','decision','typed gates','request','Jev selects skills and tools and checks whether entity extraction is needed.');
  edge('tool','loop','search','research','control','Search current supplier evidence if needed; the tool cache may avoid duplicate requests.');
  edge('retrieval','loop','recall','recall','control','Choose a scoped memory query instead of repeatedly researching known evidence.');
  edge('encode','recall','embed','query vector','data','Encode the query in the same vector space as stored passages.');
  edge('nearest','recall','vectors','HNSW candidates','data','Use the actual approximate Oracle vector index.');
  edge('select','recall','rerank','reorder candidates','data','The cross-encoder selects more relevant context.');
  if(options.reranking)edge('selectjev','recall','decision','passage selection','data','Jev ranks passages only when reranking is selected.');
  edge('facts','state','context','trip + profile','data','Read current lane facts; later corrections affect subsequent context.');
  edge('history','history','context','selected memory','data',options.workflow_recall?'Read recent conversation and prior workflow outcomes.':'Read recent conversation and current-run outcomes.');
  edge('append','loop','transcript','append full results','store','Keep every response and full tool result without compacting.');
  edge('readall','transcript','context','all previous messages','data','The next turn receives the complete transcript.');
  edge('record','loop','history','decisions + outcomes','store','Persist each tool call, success or failure and conversation event.');
  edge('cachelookup','scope','cache','stable question','control','Check eligible exact or semantic answer reuse in the lane namespace.');
  edge('cachemiss','cache','context','miss → fresh context','data','Misses continue through real model generation.');
  edge('offload','search','unpack','large payload','store','Retain full tool results behind compact references.');
  edge('archive','unpack','archive','original payload','store','Persist the original and return bounded pages by ID.');
  edge('compact','context','compact','budget / request','control','Use the actual compaction policy or selected tool.');
  edge('summary','compact','archive','summary + links','store','Preserve source turns and publish their summary reference.');
  edge('placeholder','archive','context','ID + description','data','Select a compact summary pointer; unpack detail only when needed.');
  edge('measured','claude','usage','actual API counters','event','Record input, output and provider cache reads and writes.');
  edge('decisionusage','decision','usage','Jev billed tokens','event','Include real decision-model usage in API cost.');
  edge('searchusage','search','usage','Tavily credits','event','Account for actual search calls, excluding tool-cache hits.');
  edge('compare','usage','charts','observed rows','event','Store real per-turn measurements and render them for comparison.');
  function run(id,title,ids){return{id,title,blurb:'Diagram simulation of '+AGENT_INFO[agent][0]+'. Live experiments make actual provider calls.',steps:ids.filter(f=>flows.some(x=>x.id===f)).map(f=>({flow:f,note:flows.find(x=>x.id===f).note}))};}
  const runs=[run('turn',naive?'Append-only turn':native?'Native MemAgent turn':'Memory-aware turn',['ask','seed','load','facts','history','readall','begin','route','reason','tool','offload','archive','record','append','measured','searchusage','decisionusage','compare'])];
  if(!naive){runs.push(run('retrieval','Recall and select context',['ask','load','begin','retrieval','encode','nearest',...(decisions&&options.reranking?['selectjev']:['select']),'reason','record','measured','compare']));
    if(options.compaction)runs.push(run('compaction','Compact, preserve and recall',['load','history','compact','summary','placeholder','begin','reason','archive','record','measured','compare']));
    if(options.normal_cache||options.semantic_cache||options.prompt_cache)runs.push(run('cache','Observe answer and prompt reuse',['ask','cachelookup','cachemiss','begin','reason','measured','compare']));
  }
  return{tiers:tiers.filter(t=>components.some(c=>c.tier===t.id)),components,flows,runs};
}
function renderArchitecture() {
  let agent=localStorage.getItem('architecture-agent')||'custom', player=null, closed=false;
  if(!AGENT_INFO[agent])agent='custom';
  document.querySelector('#stage').innerHTML=`<div class="hero"><span class="eyebrow">REFERENCE ARCHITECTURE · THE TOKENOMICS AGENTS</span><h1>Inside each <em>agent.</em></h1><p>Choose the agent you compare in Tokenomics. Explore its actual components and step through its execution.</p></div><section class="panel"><div class="panel-body"><label for="architecture-agent">Agent architecture</label><select id="architecture-agent">${Object.entries(AGENT_INFO).map(([k,v])=>`<option value="${k}" ${k===agent?'selected':''}>${v[0]}</option>`).join('')}</select><p class="hint" id="architecture-description"></p><div id="architecture-options" class="option-grid"></div><div id="architecture-badges" class="hero-badges"></div></div></section><div id="architecture-root"></div>`;
  function draw(){
    if(closed)return;player?.reset();
    const options=agent==='naive'?Object.fromEntries(Object.keys(OPTION_INFO).map(k=>[k,false])):agent==='memorizz'?{...selectedExperimentOptions(),...Object.fromEntries(Object.keys(OPTION_INFO).map(k=>[k,k!=='reranking']))}:selectedExperimentOptions();
    $('#architecture-description').textContent=AGENT_INFO[agent][1];
    $('#architecture-options').innerHTML=['custom','decisions'].includes(agent)?Object.entries(OPTION_INFO).map(([k,[label,hint]])=>`<label class="option-hint"><span><input type="checkbox" data-architecture-option="${k}" ${options[k]?'checked':''}> ${label}</span><small>${hint}</small></label>`).join(''):'';
    document.querySelectorAll('[data-architecture-option]').forEach(input=>input.onchange=()=>{options[input.dataset.architectureOption]=input.checked;localStorage.setItem('tokenomics-options',JSON.stringify(options));draw();});
    const spec=agentArchitecture(agent,options);
    $('#architecture-badges').innerHTML=`<span class="pill good">${spec.components.length} components</span><span class="pill">${spec.runs.length} execution scenarios</span><span class="pill warn">Diagram simulation</span><a class="pill" href="#tokenomics">Run these agents live ↗</a>`;
    player=RefArch.render($('#architecture-root'),spec);
  }
  $('#architecture-agent').onchange=e=>{agent=e.target.value;localStorage.setItem('architecture-agent',agent);draw();};draw();
  return{reset(){closed=true;player?.reset();}};
}
