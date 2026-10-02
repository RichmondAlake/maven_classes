"""Visible replacements used to build the Memorizz edition from the lesson."""


def memory_namespace(kind):
    """Separate knowledge kinds and embedding revisions at the provider boundary."""
    return f"travel:{EMBEDDING_IDENTITY}:{kind}"


def token_chunks(text, **kwargs):
    """Keep each knowledge passage inside the local encoder's token window."""
    tokens = embedder.tokenizer.encode(text, add_special_tokens=False, verbose=False)
    return [embedder.tokenizer.decode(tokens[start:start + 160])
            for start in range(0, len(tokens), 136)]


class LocalTravelEmbeddings(BaseEmbeddingProvider):
    """Adapt the same encoder; long structured units use normalized chunk means."""
    def get_dimensions(self):
        return DIM

    def get_default_model(self):
        return EMBEDDING_MODEL_ID

    def get_provider_info(self):
        return {"provider": "local", "model": EMBEDDING_MODEL_ID,
                "dimensions": DIM, "identity": EMBEDDING_IDENTITY}

    def get_embedding(self, text, **kwargs):
        chunks = token_chunks(text)
        if not chunks:
            raise ValueError("Cannot embed empty memory.")

        # A long workflow is searchable without silently truncating its tail.
        vector = embed(chunks).mean(axis=0)
        return (vector / np.linalg.norm(vector)).tolist()


def store_text(text, kind, metadata=None, memory_id=None, scope=SCOPE):
    """Ingest actual snippets using KnowledgeBase and native provider scope."""
    memory_id = memory_id or uuid.uuid4().hex
    existing = rows("""
        SELECT source_id FROM knowledge_base
        WHERE source_id = :id AND memory_id = :owner
          AND user_id = :owner AND namespace = :namespace
        FETCH FIRST 1 ROW ONLY
    """, {"id": memory_id, "owner": scope.owner, "namespace": memory_namespace(kind)})
    if existing:
        return memory_id

    knowledge.ingest_knowledge(
        text,
        namespace=memory_namespace(kind),
        chunking_strategy=token_chunks,
        user_id=scope.owner,
        metadata={
            "memory_id": scope.owner,
            "source_id": memory_id,
            "metadata": {
                **(metadata or {}),
                "memory_id": memory_id,
                "owner": scope.owner,
                "thread": scope.thread_id,
                "kind": kind,
                "embedding_identity": EMBEDDING_IDENTITY,
            },
        },
        embeddings="required",
    )
    return memory_id


def retrieve(question, kind="policy", k=6, scope=SCOPE):
    """MemoryManager calls the Oracle provider's opted-in HNSW retrieval."""
    if not isinstance(k, int) or not 1 <= k <= 20:
        raise ValueError("Choose k between 1 and 20.")

    if kind in {"conversation", "workflow", "summary"}:
        return retrieve_native_memory(question, kind, k, scope)

    found = manager.retrieve_relevant_memories(
        question,
        MemoryType.KNOWLEDGE_BASE,
        scope.owner,
        limit=k,
        user_id=scope.owner,
        namespace=memory_namespace(kind),
    )
    return [{
        "text": record["content"],
        "metadata": {**record.get("metadata", {}), "chunk": record["chunk_index"]},
        "distance": 1 - float(record["score"]),
    } for record in found]


def retrieve_exact(question, kind="policy", k=6, scope=SCOPE):
    """An explicit exact baseline reads the same native knowledge population."""
    found = rows("""
        SELECT content AS text, metadata, chunk_index,
               VECTOR_DISTANCE(embedding, :vector, COSINE) AS distance
        FROM knowledge_base
        WHERE memory_id = :owner AND user_id = :owner AND namespace = :namespace
        ORDER BY distance
        FETCH EXACT FIRST :count ROWS ONLY
    """, {
        "vector": array.array("f", embeddings.embed_query(question)),
        "owner": scope.owner,
        "namespace": memory_namespace(kind),
        "count": k,
    })
    return [{**record, "metadata": {
        **decode_json(record["metadata"]), "chunk": record["chunk_index"]
    }} for record in found]


def append_event(kind, payload, status="success", run_id=None, step=None, scope=SCOPE):
    """Persist real conversation units and one Workflow for each execution step."""
    record = {
        "run_id": run_id,
        "step_no": step,
        "status": status,
        "payload": payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if kind == "conversation":
        unit = ConversationMemoryUnit(
            role=payload["role"],
            content=pretty(record),
            timestamp=record["created_at"],
            memory_id=scope.owner,
            thread_id=scope.thread_id,
            user_id=scope.owner,
        )
        return provider.store(memory_unit=unit, memory_id=scope.owner)

    if kind == "workflow":
        unit = Workflow(
            name=f"Agent iteration {step}",
            description=f"{payload['decision']['action']} returned {status}",
            steps=record,
            memory_id=f"workflow:{scope.owner}:{scope.thread_id}",
            user_id=scope.owner,
            outcome=WorkflowOutcome.SUCCESS if status == "success" else WorkflowOutcome.FAILURE,
        )
        return provider.store(unit.to_dict(), memory_store_type=MemoryType.WORKFLOW_MEMORY)

    # Typed entity audit events remain in the explicit host provenance ledger.
    return host_append_event(kind, payload, status, run_id, step, scope)


def recent_events(kind, limit=6, active_only=False, scope=SCOPE):
    """Read native conversations; read ordered scoped workflow steps relationally."""
    if kind == "conversation":
        units = provider.retrieve_conversation_history_ordered_by_timestamp(
            scope.owner,
            user_id=scope.owner,
            thread_id=scope.thread_id,
        )
        records = [{"event_id": unit["_id"], **decode_json(unit["content"])}
                   for unit in units if not active_only or not unit.get("summary_id")]
        return records[-limit:]

    if kind == "workflow":
        units = rows("""
            SELECT workflow_id AS event_id, steps
            FROM workflow_memory
            WHERE memory_id = :memory AND user_id = :owner
            ORDER BY created_at DESC, workflow_id DESC
            FETCH FIRST :count ROWS ONLY
        """, {"memory": f"workflow:{scope.owner}:{scope.thread_id}",
              "owner": scope.owner, "count": limit})
        return [{"event_id": unit["event_id"], **decode_json(unit["steps"])}
                for unit in reversed(units)]

    return host_recent_events(kind, limit, active_only, scope)


def sync_entity_profile(scope=SCOPE):
    """Publish host-validated current facts through Memorizz EntityMemory."""
    profile = read_state("profile", "__profile__", scope)
    if not profile:
        return

    attributes = [{
        "name": name,
        "value": json.dumps(fact["value"]),
        "source": fact["source_turn_id"],
        "confidence": 1.0,
    } for name, fact in profile["state"].items()]
    entity_memory.upsert_entity(
        name="traveler",
        entity_type="travel_preferences",
        identity_key="authenticated_user",
        memory_id=scope.owner,
        user_id=scope.owner,
        attributes=attributes,
    )


def decode_entity_value(value):
    """Accept native string attributes and the host bridge's JSON-encoded values."""
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # MemAgent's entity tools can save an ordinary string such as window.
        return value


def load_profile(scope=SCOPE):
    """Read the framework's current entity facts, including source-turn provenance."""
    entity = entity_memory.get_entity_by_name("traveler", memory_id=scope.owner, user_id=scope.owner)
    if not entity:
        return {}
    return {attribute["name"]: {
        "value": decode_entity_value(attribute["value"]),
        "source_turn_id": attribute.get("source"),
    } for attribute in entity.get("attributes", [])}


def memory_pointers(scope=SCOPE, limit=6):
    """Only expose summary archives after native source linking succeeded."""
    return rows("""
        SELECT memory_id, kind, description, created_at
        FROM AM_BLOBS_V2 b
        WHERE owner_id = :owner AND thread_id = :thread
          AND (kind <> 'summary' OR EXISTS (
              SELECT 1 FROM summaries s
              WHERE s.summary_id = b.memory_id AND s.user_id = :owner
          ))
        ORDER BY created_at DESC
        FETCH FIRST :count ROWS ONLY
    """, {"owner": scope.owner, "thread": scope.thread_id, "count": limit})


def compact_thread(keep=2, quality_gate=None, scope=SCOPE):
    """Summarize real turns; Memorizz atomically links and marks native sources."""
    turns = recent_events("conversation", limit=100, active_only=True, scope=scope)
    older = turns[:-keep] if keep else turns
    if not older:
        return {"compacted": 0, "description": "No older active turns."}

    summary = draft_summary(older, scope)
    if quality_gate and not quality_gate(summary, older):
        raise ValueError("Summary quality gate rejected compaction.")

    memory_id = save_blob("summary", summary["description"], {"summary": summary, "source_turns": older}, scope)
    try:
        provider.store_summary_with_links({
            "summary_id": memory_id,
            "content": pretty(summary),
            "source_message_ids": [turn["event_id"] for turn in older],
            "memory_id": scope.owner,
            "user_id": scope.owner,
            "thread_id": scope.thread_id,
            "memory_units_count": len(older),
            "embedding": embedding_provider.get_embedding(summary["summary"]),
        })
    except Exception:
        # A failed linking transaction must not publish an active summary pointer.
        execute("DELETE FROM AM_BLOBS_V2 WHERE memory_id = :id AND owner_id = :owner",
                {"id": memory_id, "owner": scope.owner})
        conn.commit()
        raise

    store_text(summary["summary"], "summary", {"archive_id": memory_id,
               "description": summary["description"]}, memory_id, scope)
    return {"memory_id": memory_id, "description": summary["description"], "compacted": len(older)}


def hnsw_plan(question, scope=SCOPE):
    """Verify the optimizer uses the index on Memorizz's native knowledge table."""
    execute("""
        EXPLAIN PLAN SET STATEMENT_ID = 'MZ_HNSW' FOR
        SELECT /*+ VECTOR_INDEX_TRANSFORM(knowledge_base) */ content,
               VECTOR_DISTANCE(embedding, :vector, COSINE) AS distance
        FROM knowledge_base
        WHERE memory_id = :owner AND user_id = :owner AND namespace = :namespace
        ORDER BY distance
        FETCH APPROX FIRST 6 ROWS ONLY
    """, {"vector": array.array("f", embeddings.embed_query(question)),
          "owner": scope.owner, "namespace": memory_namespace("policy")}, cache_statement=False)
    return "\n".join(record["plan_table_output"] for record in rows("""
        SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY(NULL, 'MZ_HNSW', 'BASIC'))
    """))


def retrieve_native_memory(question, kind, k=6, scope=SCOPE):
    """Use Memorizz's native stores for selective recall, with thread boundaries."""
    memory_types = {
        "conversation": MemoryType.CONVERSATION_MEMORY,
        "workflow": MemoryType.WORKFLOW_MEMORY,
        "summary": MemoryType.SUMMARIES,
    }
    memory_id = f"workflow:{scope.owner}:{scope.thread_id}" if kind == "workflow" else scope.owner

    # Workflow rows encode the thread in memory_id; the other stores have a thread_id column.
    thread_filter = {} if kind == "workflow" else {"thread_id": scope.thread_id}

    found = manager.retrieve_relevant_memories(
        question, memory_types[kind], memory_id,
        limit=k, user_id=scope.owner, **thread_filter,
    )
    result = []
    for record in found:
        if kind == "workflow":
            text = pretty(record["steps"])
            source_id = record["workflow_id"]
            description = record.get("description")
        elif kind == "summary":
            summary = decode_json(record["content"])
            text = summary["summary"]
            source_id = record["summary_id"]
            description = summary["description"]
        else:
            event = decode_json(record["content"])
            text = event["payload"]["content"]
            source_id = record["_id"]
            description = event["payload"]["role"]

        result.append({
            "text": text,
            "distance": 1 - float(record["score"]),
            "metadata": {"memory_id": source_id, "chunk": 0, "kind": kind,
                         "owner": scope.owner, "thread": scope.thread_id,
                         "description": description},
        })
    return result
