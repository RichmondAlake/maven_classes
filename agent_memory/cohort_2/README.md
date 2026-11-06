# Implementing Memory Aware Agents · Cohort 2

Build a personal travel research assistant one Part at a time with Oracle AI Database, the raw Anthropic client and live Tavily search. The learner supplies trip details and preferences. Sources retain URLs and collection time; this course does not invent supplier inventory, prices or reservation receipts. Booking requires a supplier API beyond Tavily web search.

Cohort 2 is one four-hour session that teaches from one notebook, carried forward from Cohort 1. Cohort 1's materials stay unchanged in [`../cohort_1`](../cohort_1/README.md).

Cohort 2 adds **Toolbox Memory** (Part 8 in the from-scratch notebook, Part 9 in the Memorizz edition). The from-scratch notebook runs Parts 0–11 and the Memorizz edition Parts 0–12. The from-scratch notebook defines its agent's action protocol in Part 11; the Memorizz edition keeps a Part 5 · Agentic RAG, where MemAgent first chooses its own tools. Embeddings, HNSW retrieval, reranking, RAG, Agentic RAG, session memory, semantic caching, conversational memory, toolbox memory, workflow memory, and compaction/compression build into an explicit agent loop.

| Notebook | What it teaches |
|---|---|
| [From scratch](implementing_memory_aware_agents.ipynb) | Explicit memory/context construction; OracleVS and OracleSemanticCache simplify vector storage while raw Anthropic handles generation |
| [Memorizz edition](implementing_memory_aware_agents_memorizz.ipynb) | The same Parts built with the published Memorizz package: MemAgent runs the agent loop; KnowledgeBase, EntityMemory, SemanticCache, Toolbox routing, workflow memory, tool-result offloading and summaries store memory in Oracle |

The session's slides are in [`presentation/`](presentation/introduction_to_ai_agent_memory_cohort_2.pdf).

Both notebooks include their actual outputs from successful live runs, including rendered tables, diagrams, provider usage and HNSW plans. Open them to read the results without running cells. Rerun them with your own credentials and database to refresh the results; the isolated database schemas used for the recorded notebook runs were removed after validation.

The [appbook](appbook/README.md) runs at http://localhost:8031 and provides live interaction interfaces: a queryable embedding space, paired reranking/cache/compaction answers, a toolbox page that ranks stored tool definitions and runs offered tools through the host guard, a context-window inspector, and three-agent Tokenomics experiments (custom, Memorizz MemAgent and append-only). Its sections 01–11 match the from-scratch notebook's Parts 1–11. Its reference architecture includes three teaching simulations. Chat and experiments use actual providers; the memory backend defaults to the published Memorizz package. The traveler selector restores saved users; **New user** starts with empty memories. Notebook prose and implementation code are kept out of the app interface.

## Run the notebooks

Use Python 3.11 or 3.12 and a dedicated Oracle AI Database schema with VECTOR support, table quota, and a configured HNSW vector-memory pool. Do not use an administrative schema. This validation uses Oracle 26ai.

Open Jupyter from the `cohort_2` folder. Each notebook starts with a `%pip install` cell that installs the requirements into its active kernel, including `sentence-transformers`. Run that cell once, restart the kernel after installation, then run the lesson cells in order. The Memorizz notebook installs `requirements-memorizz.txt`, which includes the base requirements and the published Memorizz package.

```bash
python -m pip install -r requirements.txt
# For the Memorizz edition:
python -m pip install -r requirements-memorizz.txt
jupyter lab
```

Set `ORACLE_USER` and `ORACLE_DSN` or enter them at the notebook prompts. Oracle's password and provider API keys use hidden `getpass` input. Anthropic and Tavily are mandatory; the Memorizz edition also needs a Hugging Face access token (`HF_TOKEN`) for its hosted reranker. The local MiniLM encoder needs no embedding-service key: **embedding identity** is a configuration fingerprint, not a secret. The caching course uses paid Voyage embeddings separately and explains why that key is needed there.

Run Parts in order. The first `rag_answer(question, trip=TRIP, ...)` receives the explicit trip extracted from your own request; the model has no implicit knowledge of a Lisbon trip. There is no generation-off mode or canned replacement for a failed model call.

Interactive runs request keys even when environment credentials exist. `NOTEBOOK_USE_ENV_KEYS=1` is an explicit credential-entry option for unattended validation and the app server; it never switches generation off. No API key is saved in notebook cells or observed outputs.

## Memory and context behavior

HNSW is the actual retrieval path. An exact nearest-neighbor baseline and an Oracle execution-plan inspection make the distinction observable. Reranking uses a pandas before/after table and preserves source provenance.

Every iteration reads current trip state, current preferences, bounded active conversation, recent workflow decisions/outcomes, and memory placeholders. Workflow memory records the action and its success/failure; it is not a seeded procedure or a simulated booking receipt. The static policy/tool prefix has an Anthropic cache breakpoint; changing memory follows it. Inspect provider read/write token counts and the exact per-iteration context in the notebook and appbook.

Toolbox memory stores tool definitions, not code. Five real tools that need no API key (Open-Meteo weather and daylight, Frankfurter ECB exchange rates, Nager.Date public holidays, and a time-zone difference) are registered from their Python signatures and docstrings, embedded, and stored in Oracle. Each iteration offers the tools retrieved for the request within a distance limit, plus any the agent found with `find_tools`. They sit after the cache breakpoint, so a different tool list does not change the cached prefix. `use_tool` runs only a function the host registered and offered in that context, with exactly its parameters.

Full Tavily payloads are offloaded to scoped, checksum-verified Oracle archives. Placeholders contain an ID and description. `unpack` returns one bounded page; that page stays visible in the next model invocation. The model can call `compact`, and the host separately counts real prompt tokens and enforces a threshold. Summaries retain their original sources. 
Semantic answer reuse is reserved for stable educational explanations, with owner/model/vector namespace, distance threshold and expiry envelope. Search evidence and live offers do not become confirmed quotes through caching. Source-URL membership checks preserve provenance but do not establish claim-by-claim entailment.

## Validate or rebuild

`tests/run_live_notebook.py` executes every Python cell, captures outputs/plots, uses an isolated randomly named Oracle schema, and removes only that schema afterward. Provider keys remain in process memory. Its private `--env-file` is explicitly selected and only relevant provider keys are read. Required travel arguments are learner/test inputs, not a fixture catalog.

```bash
python tests/run_live_notebook.py implementing_memory_aware_agents_memorizz.ipynb \
  --request 'YOUR TRIP REQUEST' \
  --policy-query 'YOUR SUPPLIER POLICY QUERY' \
  --live-query 'YOUR LIVE SEARCH QUERY' \
  --preference 'YOUR EXPLICIT PREFERENCE'
```

The local validation runner currently targets Docker container `acme-oracle-free`, `127.0.0.1:1521/FREEPDB1`. Change those deployment settings before using another classroom database. Credentials are never printed. The app uses its separate persistent schema.

Reviewed teaching sources are in `tools/travel_lesson.txt` (from scratch) and `tools/memorizz_lesson.txt` (Memorizz edition; its Part 9.1 tools are the from-scratch lesson's Part 8.1). Rebuild in order; `build_notebooks.py` also regenerates the appbook's `backend/notebook_core.py` from the runtime-tagged cells.

Rebuilding regenerates lesson cells without their recorded outputs. After validating a rebuilt notebook, review its generated `.executed.ipynb` and replace the matching main notebook with that file to publish the refreshed outputs. Temporary executed copies and validation reports are ignored by Git; main notebook outputs are retained.

```bash
python tools/build_notebooks.py
python tools/build_memorizz.py
```

The app imports **runtime-tagged cells**, not notebook demonstration inputs. Diagrams have embedded PNG attachments plus standalone SVG/PNG and [editable Mermaid](data/travel_assistant_flow.mmd), so images survive moving the notebook.

The recorded outputs are in the main notebooks. Small-run latency/cost measurements illustrate mechanisms; they are not production scaling or answer-quality benchmarks.

Local environments, credentials, app experiment data, logs and generated test artifacts are excluded by `.gitignore`. Course source, required diagram assets, runnable tests and dependency files remain publishable.
