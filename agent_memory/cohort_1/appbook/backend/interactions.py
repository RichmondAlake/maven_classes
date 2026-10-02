"""Live interfaces: actual vectors, paired answers and compression measurements."""
import array
import copy
import hashlib
import json
import time
import zlib

import numpy as np

from . import runtime as rt


def settings(c):
    state = c.read_state(scope=c.SCOPE)
    return {
        "trip": state["state"]["trip"] if state else {},
        "profile": c.load_profile(c.SCOPE),
    }


def metric(c, start, seconds, **extra):
    calls = c.CALLS[start:]
    cost = (
        None
        if any(x["estimated_usd"] is None for x in calls)
        else sum(x["estimated_usd"] for x in calls)
    )
    return {
        "seconds": seconds,
        "estimated_usd": cost,
        "provider_calls": len(calls),
        "input_tokens": sum(x["input_tokens"] for x in calls),
        "cache_read_tokens": sum(x["cache_read_tokens"] for x in calls),
        "cache_write_tokens": sum(x["cache_write_tokens"] for x in calls),
        "output_tokens": sum(x["output_tokens"] for x in calls),
        **extra,
    }


def embeddings_space(query="", memory_text=""):
    c = rt.require()
    if memory_text:
        c.store_text(
            memory_text,
            "embedding_lab",
            {"description": "User-added embedding memory"},
            scope=c.SCOPE,
        )
    if rt.BACKEND == "memorizz":
        records = c.rows(
            """
            SELECT RAWTOHEX(id) AS id, content AS text, embedding, metadata
            FROM knowledge_base
            WHERE memory_id = :owner AND user_id = :owner AND embedding IS NOT NULL
            ORDER BY created_at DESC FETCH FIRST 80 ROWS ONLY
        """,
            {"owner": c.SCOPE.owner},
        )
    else:
        records = c.rows(
            """
            SELECT id, text, embedding, metadata FROM AM_MEMORY_V2
            WHERE JSON_VALUE(metadata, '$.owner') = :owner AND embedding IS NOT NULL
            FETCH FIRST 80 ROWS ONLY
        """,
            {"owner": c.SCOPE.owner},
        )
    points = []
    for row in records:
        vector = np.asarray(row["embedding"], dtype=float)
        meta = c.decode_json(row["metadata"])
        points.append(
            {
                "id": str(row["id"]),
                "text": row["text"],
                "vector": vector.tolist(),
                "url": meta.get("url"),
                "kind": meta.get("kind", "memory"),
            }
        )
    # Fit the view to stored vectors only. Distances always use the full vectors.
    if points:
        matrix = np.asarray([p["vector"] for p in points])
        center = matrix.mean(axis=0)
        _, _, vt = np.linalg.svd(matrix - center, full_matrices=False)
        basis = vt[:3].T
        for p in points:
            coordinates = (np.asarray(p["vector"]) - center) @ basis
            p["position"] = np.pad(coordinates, (0, 3 - len(coordinates))).tolist()
    else:
        center = np.zeros(c.DIMENSIONS)
        basis = np.eye(c.DIMENSIONS, 3)
    query_point = None
    if query:
        q = c.embed([query])[0]
        coordinates = (q - center) @ basis
        query_point = {
            "text": query,
            "vector": q.tolist(),
            "position": np.pad(coordinates, (0, 3 - len(coordinates))).tolist(),
        }
        for p in points:
            v = np.asarray(p["vector"])
            p["cosine_distance"] = float(
                1 - np.dot(v, q) / (np.linalg.norm(v) * np.linalg.norm(q))
            )
        points.sort(key=lambda p: p["cosine_distance"])
    return {
        "points": points,
        "query": query_point,
        "dimensions": c.DIMENSIONS,
        "projection": "PCA",
        "model": c.EMBEDDING_MODEL_ID,
    }


def retrieval(query, fetch_web=False, inspect_plan=True):
    c = rt.require()
    if fetch_web:
        c.ingest_search(c.web_search(query), scope=c.SCOPE)
    started = time.perf_counter()
    candidates = c.retrieve(query, scope=c.SCOPE)
    seconds = time.perf_counter() - started
    return {
        "query": query,
        "candidates": candidates,
        "seconds": seconds,
        "plan": c.hnsw_plan(query, c.SCOPE) if inspect_plan else None,
        "index_used": "HNSW",
    }


def evidence_answer(c, query, passage, purpose):
    context = {**settings(c), "question": query, "evidence": [passage]}
    started, call_start = time.perf_counter(), len(c.CALLS)
    answer = c.llm_text(c.pretty(context), system=c.SYSTEM, purpose=purpose)
    return {
        "answer": answer,
        "context": passage,
        "metrics": metric(c, call_start, time.perf_counter() - started),
    }


def reranking(query, fetch_web=False):
    c = rt.require()
    found = retrieval(query, fetch_web, inspect_plan=False)
    candidates = found["candidates"]
    if not candidates:
        raise ValueError(
            "No stored passages match this query. Search the live web and store evidence first."
        )
    started = time.perf_counter()
    reranked = c.rerank(query, candidates, keep=len(candidates))
    rerank_seconds = time.perf_counter() - started
    left = evidence_answer(c, query, candidates[0], "comparison: vector top 1")
    right = evidence_answer(c, query, reranked[0], "comparison: reranked top 1")
    left["metrics"]["seconds"] += found["seconds"]
    right["metrics"]["seconds"] += found["seconds"] + rerank_seconds
    return {
        "query": query,
        "without": left,
        "with": right,
        "candidates": candidates,
        "reranked": reranked,
        "rerank_seconds": rerank_seconds,
        "same_passage": candidates[0]["text"] == reranked[0]["text"],
    }


def semantic_comparison(query, cache_id):
    from .tokenomics import educational

    if not educational(query):
        raise ValueError("Use a stable agent-memory concept for this cache comparison.")
    c = rt.require()
    scope = rt.replace(
        c.SCOPE, user_id=c.SCOPE.user_id + ":cache-comparison:" + cache_id
    )
    system = "Explain the requested agent-memory concept. Do not research current travel offers."
    start, calls = time.perf_counter(), len(c.CALLS)
    direct = c.llm_text(query, system=system, purpose="comparison: cache disabled")
    left = {
        "answer": direct,
        "cache_hit": False,
        "metrics": metric(c, calls, time.perf_counter() - start),
    }
    start, calls = time.perf_counter(), len(c.CALLS)
    rt.register_cache_scope(c.cache_namespace(query, scope))
    cached = c.cached_explanation(query, scope=scope)
    right = {**cached, "metrics": metric(c, calls, time.perf_counter() - start)}
    return {"query": query, "without": left, "with": right, "cache_id": cache_id}


def compaction_comparison(query):
    c = rt.require()
    turns = c.recent_events("conversation", limit=60, active_only=True, scope=c.SCOPE)
    if len(turns) < 2:
        raise ValueError(
            "Have at least two conversation turns with the travel assistant first."
        )
    older, newest = turns[:-1], turns[-1:]
    start, calls = time.perf_counter(), len(c.CALLS)
    summary = c.draft_summary(older, c.SCOPE)
    summary_metrics = metric(c, calls, time.perf_counter() - start)
    raw = c.pretty(older).encode("utf-8")
    archive = zlib.compress(raw)
    assert zlib.decompress(archive) == raw
    full_context = {**settings(c), "question": query, "conversation": turns}
    compact_context = {
        **settings(c),
        "question": query,
        "earlier_summary": summary["summary"],
        "conversation": newest,
    }

    def answer(context, purpose):
        count = c.client.messages.count_tokens(
            model=c.LLM_MODEL_ID,
            system=c.SYSTEM,
            messages=[{"role": "user", "content": c.pretty(context)}],
        ).input_tokens
        start, calls = time.perf_counter(), len(c.CALLS)
        text = c.llm_text(c.pretty(context), system=c.SYSTEM, purpose=purpose)
        return {
            "answer": text,
            "context_tokens": count,
            "metrics": metric(c, calls, time.perf_counter() - start),
        }

    left = answer(full_context, "comparison: complete history")
    right = answer(compact_context, "comparison: compacted history")
    return {
        "query": query,
        "without": left,
        "with": right,
        "turns": [
            {"role": t["payload"]["role"], "content": t["payload"]["content"]}
            for t in turns
        ],
        "summary": summary,
        "summary_metrics": summary_metrics,
        "archive": {
            "original_bytes": len(raw),
            "compressed_bytes": len(archive),
            "bytes_saved": len(raw) - len(archive),
            "lossless_verified": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "tokens_saved": left["context_tokens"] - right["context_tokens"],
        "active_history_modified": False,
    }


def compact_now():
    """Apply the actual history policy and expose its persisted summary and sources."""
    c = rt.require()
    before_turns = c.recent_events("conversation", limit=100, active_only=True, scope=c.SCOPE)
    original_ids = [t["event_id"] for t in before_turns]
    before_context = c.build_context("Inspect compacted conversation.", "compaction_preview", c.SCOPE)
    before_tokens = c.context_tokens(before_context)
    started, calls = time.perf_counter(), len(c.CALLS)
    result = c.compact_thread(keep=2, scope=c.SCOPE)
    after_context = c.build_context("Inspect compacted conversation.", "compaction_preview", c.SCOPE)
    after_tokens = c.context_tokens(after_context)
    originals = c.recent_events("conversation", limit=100, scope=c.SCOPE)
    retained = {t["event_id"] for t in originals}
    archive = c.load_blob(result["memory_id"], c.SCOPE) if result.get("memory_id") else None
    return {**result, "trigger": "button", "before_tokens": before_tokens, "after_tokens": after_tokens,
            "before_active_turns": len(before_turns),
            "after_active_turns": len(c.recent_events("conversation", limit=100, active_only=True, scope=c.SCOPE)),
            "originals_preserved": set(original_ids) <= retained,
            "summary": archive["summary"] if archive else None,
            "source_turn_ids": original_ids[:-2] if result.get("compacted") else [],
            "pointers": c.memory_pointers(c.SCOPE),
            "metrics": metric(c, calls, time.perf_counter() - started)}
