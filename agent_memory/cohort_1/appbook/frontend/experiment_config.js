"use strict";
const AGENT_INFO = {
  custom: ['Custom memory agent', 'A manual agent loop builds each prompt from Oracle trip, entity, conversation and workflow memory. Uses the options below.'],
  memorizz: ['Memorizz MemAgent', 'The native Memorizz loop manages recall, compaction and tool offloading through its Oracle provider. Uses a fixed optimized preset.'],
  naive: ['Append-only agent', 'Claude receives the starting trip and every previous message and full tool result. No memory recall, compaction or caching.'],
  decisions: ['Custom + Jev decisions', 'The custom loop adds hosted Jev routing, entity detection, passage selection and summary quality decisions. Jev usage is included in cost.'],
};
const OPTION_INFO = {
  prompt_cache: ['Provider prompt caching', 'Anthropic reuses the stable instruction prefix. Cache reads and writes come from the provider response.'],
  normal_cache: ['Exact answer cache', 'Reuse an identical stable educational request within the same trip and profile. A hit avoids generation.'],
  embedding_cache: ['Embedding cache', 'Reuse vectors for identical text. Saves local encoder work; this encoder has no embedding API charge.'],
  semantic_cache: ['Semantic answer cache', 'Oracle vector lookup reuses a sufficiently similar stable explanation. Live travel answers bypass this cache.'],
  tool_cache: ['Tavily tool cache', 'Reuse the same search request for 60 seconds. Cache hits avoid another Tavily API call.'],
  reranking: ['Reranking', 'Reorder HNSW candidates for the query before selecting context. Jev performs selection in the decision agent.'],
  offloading: ['Tool result offloading', 'Store full tool payloads and send compact references. Read needed details by ID in bounded pages.'],
  compaction: ['Context compaction', 'Summarize older conversation when the context budget is reached. Originals remain available through references.'],
  workflow_recall: ['Prior workflow recall', 'Bring earlier decisions and tool outcomes into the next loop to avoid repeating completed work.'],
};
const SCENARIO_INFO = {
  research: ['Trip research and follow-ups', 'Research the active trip, then refine flights, hotels, transport and supplier policies across distinct turns.'],
  preferences: ['Preference changes and recall', 'Introduce and correct preferences in isolated experiment memory, then ask follow-ups that require remembering them.'],
  context: ['Growing conversation', 'Build a detailed trip discussion to observe increasing context and the effect of compaction and offloading.'],
  cache: ['Deliberate cache reuse', 'Repeat and paraphrase stable memory questions on purpose to compare exact, semantic and provider cache hits.'],
};
function selectedExperimentOptions() {
  try { return Object.fromEntries(Object.keys(OPTION_INFO).map(k => [k, JSON.parse(localStorage.getItem('tokenomics-options') || '{}')[k] !== false])); }
  catch (_) { return Object.fromEntries(Object.keys(OPTION_INFO).map(k => [k, true])); }
}
