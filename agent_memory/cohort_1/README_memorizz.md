# Memorizz equivalent · published package

[Open the notebook](implementing_memory_aware_agents_memorizz.ipynb). It preserves Parts 0–12 of the revised travel lesson and uses `memorizz[oracle,anthropic]==0.13.0` from PyPI.

Install `requirements-memorizz.txt` in the kernel. The notebook reports the installed package version and path and rejects editable checkouts. No local source path is injected. Generation always uses the raw Anthropic client; Tavily supplies live search evidence. API keys use getpass.

| Component | Actual implementation |
|---|---|
| Source ingestion | KnowledgeBase with bounded encoder-token chunks and source metadata |
| Selective recall | MemoryManager with owner, namespace and thread boundaries |
| Approximate retrieval | OracleProvider `index_policy="lazy", vector_search_mode="approximate"`; actual HNSW plan inspection |
| Conversation | Native ConversationMemoryUnit records, read by provider history calls |
| Current preferences | EntityMemory with typed host validation and source-turn provenance |
| Workflow | One Workflow unit for each actual decision/tool outcome, including failures |
| Compaction | `store_summary_with_links` atomically links and marks native source turns |
| Semantic answer cache | Requested langchain_oracledb OracleSemanticCache around raw Anthropic generation |
| Session and archives | Explicit versioned host state and scoped compressed/checksummed blobs |

The published Oracle provider supports explicit approximate vector search, native workflow/summary scopes and atomic summary links. Exact search remains its default, so this lesson explicitly selects approximate HNSW retrieval.

The current-fact bridge first updates versioned host state, then publishes EntityMemory attributes. Those two writes are not a distributed atomic transaction. For summaries, the archive is written before native linking; only successfully linked summaries appear as placeholders, and failed linking removes the unpublished archive. These boundaries are stated in the notebook.

Entity reads accept both JSON-encoded bridge values and plain string values saved by native MemAgent tools. Profile questions leave unknown trip fields empty; they do not fabricate an active trip.

The appbook defaults to this backend, shares runtime-tagged notebook code, and exposes actual context/usage/rows. Use `--memory-backend scratch` to run the from-scratch edition. The notebook keeps the manual loop visible; Tokenomics also runs native MemAgent with its own framework context and tool policies.

Validation reports record source fingerprints, all executed cells, actual provider usage and an HNSW execution plan. The course uses actual trip requests and web results; it contains no fabricated inventory or booking receipts.
