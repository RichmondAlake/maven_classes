# Wayfinder · memory-aware travel appbook

Open http://localhost:8031. Travel chat uses raw Claude Opus 5.5, live Tavily search and durable Oracle memory. The app presents interaction interfaces; it has no setup section, notebook downloads or implementation-code panels.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python sync_notebook.py
python server.py --memory-backend memorizz
```

The server requests missing Anthropic and Tavily keys with getpass. `--ask-keys` always prompts; `--env-file PATH` reads an explicitly selected private file. Add `TYPESAFE_API_KEY` to the server environment or that file to enable the Jev comparison agent. Provider keys stay in process memory. The server binds to loopback.

Set `ORACLE_USER`, `ORACLE_PASSWORD` and `ORACLE_DSN` for a dedicated Oracle AI Database schema with HNSW vector memory. `provision_local.py --container YOUR_ORACLE_CONTAINER --dsn YOUR_DSN` creates a randomly named schema and writes its database credentials to ignored `data/oracle_v2.json` with mode 0600. Existing schemas and the prior `oracle.json` are preserved. Memorizz 0.13.0 is installed from PyPI in the appbook environment. No source checkout is added to Python imports.

The Travel Assistant header has a **New conversation** button. It clears the active conversation by switching to a fresh durable thread. The dialog lets you keep the current trip; remembered preferences carry over. Saving a trip updates the current thread.

Sections 01–12 provide live interfaces:

- Embeddings: real Oracle vectors and a live query, an orbitable 3D PCA projection, all 384 dimensions shown as heatmaps, and an animated dot-product calculation. Cosine distance uses the complete vectors rather than projected coordinates.
- Retrieval: actual Oracle HNSW search, with optional Tavily ingestion of fresh evidence.
- Reranking: one query and candidate pool, two chat panes using vector top 1 or cross-encoder top 1. Rankings and selected passages are visible; identical selections are reported honestly.
- RAG and agentic retrieval: sourced answers and the agent's actual decisions and tool outcomes.
- Profile: explicit names persist in native EntityMemory with their source turn and appear in the active trip card across new conversations.
- Session, conversational and workflow memory: durable trip state, preference extraction/correction, remembered turns, and execution outcomes.
- Semantic cache: paired fresh generation and Oracle semantic-cache reuse, with measured cold misses and warm hits. An empty-cache control selects a new isolated namespace. Reuse is limited to stable agent-memory questions.
- Compaction/compression: answers using full history or an actual LLM summary, counted context tokens, summary-generation overhead, and lossless archive storage savings. The paired comparison preserves active history. The separate Auto compact conversation button applies compaction, reports measured context size and exposes the saved summary and linked originals. Small conversations may not shrink.
- Prompt caching: provider-reported uncached input, cache writes, cache reads and output, with actual API cost estimates.

The reference architecture offers four selectable diagrams matching the Tokenomics implementations. Custom/decision diagrams share optimization settings with Tokenomics; the native Memorizz preset and append-only path remain distinct. Each diagram has component inspection and execution simulations. The context inspector displays the stable prompt prefix, changing trip/profile/conversation/workflow memory, archive placeholders and actual per-invocation budgets. The Oracle explorer shows real persisted rows.

## Tokenomics

Choose a trip-research, preference-correction, growing-context or deliberate-cache scenario; generate and review evolving turns with raw Claude; or write an exact custom sequence. Choose 1–30 turns per agent and compare four actual implementations:

| Agent | Implementation |
|---|---|
| Custom memory agent | Explicit agent loop with Oracle retrieval, memory and context construction |
| Memorizz MemAgent | Published-package MemAgent using the native Oracle provider, bounded recall, entity memory, compaction and tool-result offloading |
| Append-only agent | Raw Claude loop retaining every conversation message and complete search result |
| Custom + Jev decisions | Custom loop with actual Jev entity, routing and summary-quality decisions |

Custom and Jev agents expose nine switches: provider prompt cache, exact answer cache, embedding cache, semantic cache, Tavily tool cache, reranking, tool-result offloading, context compaction and prior workflow recall. Memorizz uses its optimized preset. Answer caches serve stable educational requests; live travel answers are regenerated.

All agents receive the same prompts, initial trip and preferences, with separate histories/cache scopes. Order rotates between turns. The default memory-agent context budget is 24,000 tokens; very small budgets can fail when required instructions, tools and current evidence cannot fit. Native archive expansion reads bounded pages rather than reloading an entire search payload.

Click each legend label to hide or restore that agent in all four charts without changing results. Every agent and optimization selector includes a visible hint. Late start/poll responses cannot update a departed view. Charts show observed latency, cumulative estimated API cost, processed input tokens and provider cache reads. Prices were checked on 2026-10-02 against [Anthropic](https://platform.claude.com/docs/en/models/opus-5-5/overview), [Tavily](https://www.tavily.com/pricing) and [TypeSafe](https://docs.typesafe.ai/models). Cost includes initialization calls, Anthropic usage, Tavily credits and Jev input tokens. Missing usage remains unknown. Local encoder/database compute and account discounts are excluded. Failed turns retain measured costs; no answer-quality score or simulated benchmark values are supplied.

Synthetic input generation is one shared setup charge, reported separately from lane costs. Custom sequences require one request per turn; only the deliberate-cache scenario cycles queries. Answer-cache identity includes current trip and profile values so preference corrections invalidate reuse.

Progress and exportable results persist in ignored `data/tokenomics/`. Stopping an experiment finishes its current turn. Experiments do not switch the traveler's active thread or overwrite their preferences.

## Validation

Run `tests/check_interactions.py` against the live server for actual vector, HNSW, paired-answer, cache, compaction and four-agent checks. It makes paid provider calls and starts a fresh conversation with the current trip preserved. `tests/check_interaction_browser.py` checks Chromium interactions, rendered plots, paired panes, charts and mobile layout. `tests/check_live_interfaces.py` exercises the paired interfaces and conversation button through the browser with real providers. `tests/check_tokenomics_browser.py` tests starting, option submission and stopping an actual experiment. `tests/test_action_contract.py` and `tests/test_tokenomics_contract.py` check action boundaries, scope restoration, bounded archive expansion and incomplete cost accounting without initializing providers.

Reports and screenshots are generated locally and ignored by Git. Private database configuration and saved experiment files in `data/`, along with the local `.venv`, are also ignored. `sync_notebook.py` extracts runtime-tagged cells for three backend modules; it publishes no notebook prose or code to the frontend. The frontend is maintained independently. The main course notebooks retain their executed outputs.

Recent validation: `tests/check_revision_browser.py` checks all four architecture simulations, live selector hints, chart toggles, scenario previews and mobile layout. `tests/check_revision_live.py` checks HNSW, the reported reranking question, real synthetic generation and a four-agent run. `tests/check_identity_compaction_live.py` checks all three entity extractors, Jev gating, native provenance, cross-thread name recall and real compaction in an isolated thread. `tests/check_tokenomics_navigation.py` delivers an actual start response after navigation. `tests/test_workload_contract.py` checks sequencing, workload scope and profile-aware cache invalidation.

## Saved traveler profiles

The Travel assistant has a traveler selector and **New user** button. A new profile starts with empty trip, entity facts, conversation, workflow and cache scopes. The label identifies the profile in the selector; it is not automatically stored as an entity name. Switching back restores the traveler’s active trip and last conversation. The selected traveler and each last thread persist in Oracle across app restarts. **New conversation** stays within the current traveler. Tokenomics exports and the Oracle explorer follow the selected traveler.

The runtime uses the published `memorizz[oracle,anthropic]==0.13.0` wheel; `requirements-validated.txt` records the environment used for validation.

`tests/check_users_live.py` validates empty new profiles, separate Oracle rows, actual entity extraction and cross-conversation recall, and restoration of the original traveler. `tests/check_users_browser.py` exercises the profile selector and new-user dialog on desktop and mobile. `tests/test_user_isolation.py` checks experiment ownership during temporary benchmark scopes and entity values written by the host bridge or native MemAgent. Set `APPBOOK_URL` to test a separate local port.
