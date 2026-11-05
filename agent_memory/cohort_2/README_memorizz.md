# Memorizz edition · published package

[Open the notebook](implementing_memory_aware_agents_memorizz.ipynb). It teaches the same travel assistant and Parts 0–12 as the from-scratch notebook, built with `memorizz[oracle,anthropic]==0.17.23` from PyPI. Install `requirements-memorizz.txt` in the kernel. API keys use getpass.

**MemAgent** runs the agent loop and Memorizz stores every memory type in its own Oracle table. Claude runs through Memorizz's Anthropic provider (prompt caching on by default) and Tavily through its TavilyProvider.

| Part | Memorizz component |
|---|---|
| Embeddings | A local MiniLM encoder behind `BaseEmbeddingProvider` |
| Retrieval | `KnowledgeBase.ingest_knowledge` and `retrieve_knowledge_by_query` (approximate HNSW in Oracle) |
| Agentic RAG | `MemAgent` with `internet_search` and `knowledge_base_lookup` |
| Session memory | The current trip as an `EntityMemory` record |
| Semantic cache | `SemanticCache` |
| Conversational memory | MemAgent conversation memory and `capture_entity_facts` |
| Toolbox | `Toolbox.from_functions` with MemAgent semantic tool routing (`ContextPolicy`) |
| Workflow | Workflow records written by MemAgent for every turn that calls tools |
| Compaction | `ToolResultPolicy` offloading to the tool log, `generate_summaries` and `ContextPolicy` compaction |
| Integrated assistant | One MemAgent with conversation, workflow, summary, entity and knowledge memory |

Two pieces stay application code because Memorizz does not provide them: the call to the hosted Hugging Face reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`; Memorizz's built-in rerankers are Voyage, Cohere and OpenAI) and returning workflow records to the prompt. The reranker needs a Hugging Face access token (`HF_TOKEN`). The notebook says where each one is.

The appbook keeps its own Memorizz component bridge (`appbook/backend/memorizz_core.py`) for its interactive labs; its Tokenomics page runs a native MemAgent lane.
