"""Local browser adapter; every generation uses the notebook's raw Claude client."""
import importlib
import os
import threading
from dataclasses import replace
from datetime import datetime, timezone

lock = threading.RLock()
core = None
ready = False
error = None
last_run = None
activity = []
ACTIVE_SCOPE = None
DEFAULT_SCOPE = None
REGISTRY_SCOPE = None
user_runs = {}
user_activity = {}
user_calls = {}
user_searches = {}
BACKEND = os.getenv("MEMORY_BACKEND", "memorizz")


def record(name, detail):
    activity.append(
        {"time": datetime.now(timezone.utc).isoformat(), "name": name, "detail": detail}
    )
    del activity[:-80]


def users():
    c = require()
    return c.rows("""
        SELECT user_id, display_name, created_at
        FROM AM_TRAVELERS_V2 WHERE tenant_id = :tenant
        ORDER BY created_at, user_id
    """, {"tenant": DEFAULT_SCOPE.tenant_id})


def select_active(scope):
    global ACTIVE_SCOPE
    core.SCOPE = scope
    ACTIVE_SCOPE = scope


def remember_selection():
    c = require()
    selected = c.read_state("active_user", "__app__", REGISTRY_SCOPE)
    c.write_state({"user_id": ACTIVE_SCOPE.user_id}, "active_user",
                  selected["version"] if selected else None, "__app__", REGISTRY_SCOPE)


def initialize_users():
    """Register the existing traveler and restore the selected profile on restart."""
    global DEFAULT_SCOPE, REGISTRY_SCOPE
    c = core
    DEFAULT_SCOPE = c.SCOPE
    REGISTRY_SCOPE = replace(DEFAULT_SCOPE, user_id="__app_user_registry__", thread_id="__app__")
    c.create_once("""
        CREATE TABLE AM_TRAVELERS_V2 (
            tenant_id VARCHAR2(120) NOT NULL,
            user_id VARCHAR2(120) NOT NULL,
            display_name VARCHAR2(120) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
            PRIMARY KEY (tenant_id, user_id)
        )
    """)
    c.create_once("""
        CREATE TABLE AM_CACHE_SCOPES_V2 (
            namespace_hash VARCHAR2(64) PRIMARY KEY,
            owner_id VARCHAR2(64) NOT NULL
        )
    """)
    profile = c.load_profile(DEFAULT_SCOPE)
    display_name = profile.get("name", {}).get("value") or "Original traveler"
    c.execute("""
        MERGE INTO AM_TRAVELERS_V2 u
        USING (SELECT :tenant tenant_id, :user_id user_id FROM dual) s
        ON (u.tenant_id = s.tenant_id AND u.user_id = s.user_id)
        WHEN NOT MATCHED THEN INSERT (tenant_id, user_id, display_name)
        VALUES (s.tenant_id, s.user_id, :name)
    """, {"tenant": DEFAULT_SCOPE.tenant_id, "user_id": DEFAULT_SCOPE.user_id, "name": display_name})
    c.conn.commit()
    selected = c.read_state("active_user", "__app__", REGISTRY_SCOPE)
    user_id = selected["state"]["user_id"] if selected else DEFAULT_SCOPE.user_id
    exists = c.rows("SELECT user_id FROM AM_TRAVELERS_V2 WHERE tenant_id=:tenant AND user_id=:user_id",
                    {"tenant": DEFAULT_SCOPE.tenant_id, "user_id": user_id})
    scope = replace(DEFAULT_SCOPE, user_id=user_id if exists else DEFAULT_SCOPE.user_id)
    active = c.read_state("active_thread", "__app__", scope)
    if active:
        scope = replace(scope, thread_id=active["state"]["thread_id"])
    select_active(scope)


def register_cache_scope(namespace):
    """Associate the integration's opaque namespace hash with its local traveler."""
    import hashlib
    c = require()
    c.execute("""
        MERGE INTO AM_CACHE_SCOPES_V2 r
        USING (SELECT :hash namespace_hash FROM dual) s ON (r.namespace_hash=s.namespace_hash)
        WHEN NOT MATCHED THEN INSERT(namespace_hash,owner_id) VALUES(s.namespace_hash,:owner)
    """, {"hash": hashlib.sha256(namespace.encode()).hexdigest(), "owner": ACTIVE_SCOPE.owner})
    c.conn.commit()


def switch_user(user_id):
    global last_run, activity
    c = require()
    if user_id not in {user["user_id"] for user in users()}:
        raise ValueError("Choose a saved traveler profile.")
    user_runs[ACTIVE_SCOPE.owner] = last_run
    user_activity[ACTIVE_SCOPE.owner] = activity
    scope = replace(DEFAULT_SCOPE, user_id=user_id)
    active = c.read_state("active_thread", "__app__", scope)
    if active:
        scope = replace(scope, thread_id=active["state"]["thread_id"])
    select_active(scope)
    last_run = user_runs.get(scope.owner)
    activity = user_activity.setdefault(scope.owner, [])
    remember_selection()
    return snapshot()


def create_user(display_name):
    c = require()
    display_name = display_name.strip()
    if not display_name or len(display_name) > 120 or any(ord(char) < 32 for char in display_name):
        raise ValueError("Enter a profile label of 1–120 characters without control characters.")
    user_id = "traveler_" + c.uuid.uuid4().hex
    scope = replace(DEFAULT_SCOPE, user_id=user_id, thread_id="trip_" + c.uuid.uuid4().hex[:12])
    c.execute("INSERT INTO AM_TRAVELERS_V2(tenant_id,user_id,display_name) VALUES (:tenant,:user_id,:name)",
              {"tenant": scope.tenant_id, "user_id": user_id, "name": display_name})
    c.conn.commit()
    c.write_state({"thread_id": scope.thread_id}, "active_thread", thread="__app__", scope=scope)
    return switch_user(user_id)


def warm():
    global core, ready, error
    try:
        module = "memorizz_core" if BACKEND == "memorizz" else "notebook_core"
        core = importlib.import_module("." + module, package="backend")
        initialize_users()
        ready = True
        record(
            "Ready",
            "Oracle HNSW memory and raw Anthropic/Tavily clients are connected.",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:250]}"


def require():
    if not ready:
        raise RuntimeError(
            error or "The models and Oracle memory are still warming up."
        )
    return core


def snapshot():
    c = require()
    registered = users()
    return {
        "backend": BACKEND,
        "users": registered,
        "active_user": next(user for user in registered if user["user_id"] == ACTIVE_SCOPE.user_id),
        "scope": {
            "tenant": c.SCOPE.tenant_id,
            "user": c.SCOPE.user_id,
            "thread": c.SCOPE.thread_id,
        },
        "session": c.read_state(scope=c.SCOPE),
        "preferences": c.load_profile(c.SCOPE),
        "conversation": c.recent_events("conversation", limit=20, scope=c.SCOPE),
        "workflow": c.workflow_memory(20, c.SCOPE),
        "pointers": c.memory_pointers(c.SCOPE),
        "activity": activity[-15:],
        "last_run": last_run,
        "usage": user_calls.get(ACTIVE_SCOPE.owner, [])[-30:],
    }


def save_trip(request, new_thread=False):
    global last_run
    c = require()
    scope = (
        replace(c.SCOPE, thread_id="trip_" + c.uuid.uuid4().hex[:12])
        if new_thread
        else c.SCOPE
    )
    trip = c.parse_trip(request)
    c.set_trip(trip, scope)
    active = c.read_state("active_thread", "__app__", scope)
    c.write_state(
        {"thread_id": scope.thread_id},
        "active_thread",
        active["version"] if active else None,
        "__app__",
        scope,
    )
    select_active(scope)
    if new_thread:
        last_run = None
    record("Trip saved", "Dates and constraints extracted from the traveler request.")
    return snapshot()


def new_conversation(keep_trip=True):
    global last_run
    c = require()
    previous = c.read_state(scope=c.SCOPE)
    scope = replace(c.SCOPE, thread_id="trip_" + c.uuid.uuid4().hex[:12])
    if keep_trip and previous:
        c.set_trip(previous["state"]["trip"], scope)
    active = c.read_state("active_thread", "__app__", scope)
    c.write_state(
        {"thread_id": scope.thread_id},
        "active_thread",
        active["version"] if active else None,
        "__app__",
        scope,
    )
    select_active(scope)
    last_run = None
    record(
        "New conversation", "Started a fresh thread; durable preferences are retained."
    )
    return snapshot()


def chat(request):
    global last_run
    c = require()
    start, search_start = len(c.CALLS), len(c.SEARCH_CALLS)
    # A pre-loop extraction failure must not expose another user's prior trace.
    c.LAST_AGENT_RUN = {"inspections": []}
    try:
        if c.read_state(scope=c.SCOPE) is None:
            trip = c.parse_trip(request)
            # Greetings and profile facts do not create an empty active trip.
            if any(trip.get(field) for field in (
                "origin", "destination", "departure_date", "return_date", "budget"
            )):
                c.set_trip(trip, c.SCOPE)
        result = c.agent_turn(request, scope=c.SCOPE)
    except Exception as exc:
        last_run = {
            **getattr(c, "LAST_AGENT_RUN", {}),
            "calls": c.CALLS[start:],
            "prefix": c.stable_prefix(),
            "error": type(exc).__name__,
        }
        record(
            "Assistant stopped",
            "Inspect its durable workflow and retained context trace.",
        )
        user_calls.setdefault(ACTIVE_SCOPE.owner, []).extend(c.CALLS[start:])
        user_searches.setdefault(ACTIVE_SCOPE.owner, []).extend(c.SEARCH_CALLS[search_start:])
        raise
    last_run = {**result, "calls": c.CALLS[start:], "prefix": c.stable_prefix()}
    user_calls.setdefault(ACTIVE_SCOPE.owner, []).extend(c.CALLS[start:])
    user_searches.setdefault(ACTIVE_SCOPE.owner, []).extend(c.SEARCH_CALLS[search_start:])
    record(
        "Assistant answered",
        f'{len(result["inspections"])} iterations recorded with success/failure outcomes.',
    )
    return {**last_run, "responder": c.LLM_MODEL_ID, "snapshot": snapshot()}


def context(request="Inspect current trip memory."):
    c = require()
    current, budget = c.bounded_context(request, "browser_inspection", c.SCOPE)
    return {
        "prefix": c.stable_prefix(),
        "dynamic_context": current,
        "budget": budget,
        "last_run": last_run,
        "usage": user_calls.get(ACTIVE_SCOPE.owner, [])[-30:],
    }


def lab(number, payload):
    c = require()
    query = str(payload.get("query", "")).strip()
    start = len(c.CALLS)
    if number == "0":
        result = {
            "answer": c.llm_text(query or "Explain agent memory in one sentence.")
        }
    elif number == "1":
        first = payload.get("first", "I want a nonstop flight.")
        second = payload.get("second", "I prefer a direct flight.")
        vectors = c.embed([first, second])
        result = {
            "texts": [first, second],
            "dimensions": c.DIMENSIONS,
            "cosine_similarity": float(vectors[0] @ vectors[1]),
            "vector_preview": vectors[0][:12].tolist(),
        }
    elif number in {"2", "3"}:
        if not query:
            raise ValueError("Enter a supplier-policy search query.")
        search = c.web_search(query)
        c.ingest_search(search, scope=c.SCOPE)
        before = c.retrieve(query, scope=c.SCOPE)
        result = {
            "candidates": before,
            "reranked": c.rerank(query, before, keep=6),
            "plan": c.hnsw_plan(query, c.SCOPE),
            "search": search,
        }
    elif number == "4":
        session = c.read_state(scope=c.SCOPE)
        if not session:
            raise ValueError("Save your trip on the assistant page first.")
        result = c.rag_answer(
            query, session["state"]["trip"], c.load_profile(c.SCOPE), c.SCOPE
        )
    elif number in {"5", "11"}:
        return chat(query)
    elif number == "6":
        result = {
            "session": c.read_state(scope=c.SCOPE),
            "preferences": c.load_profile(c.SCOPE),
        }
    elif number == "7":
        if not query:
            raise ValueError("Enter a stable educational memory question.")
        register_cache_scope(c.cache_namespace(query, c.SCOPE))
        result = c.cached_explanation(query, scope=c.SCOPE)
    elif number == "8":
        if not query:
            raise ValueError("Enter a preference you explicitly want remembered.")
        turn = c.append_event(
            "conversation", {"role": "user", "content": query}, scope=c.SCOPE
        )
        extracted = c.extract_entities(query)
        result = {
            "extracted": extracted,
            "profile": c.update_profile(extracted, turn, c.SCOPE),
        }
    elif number == "9":
        result = {"workflow": c.workflow_memory(20, c.SCOPE)}
    elif number == "10":
        if payload.get("memory_id"):
            result = c.unpack_memory(
                payload["memory_id"],
                offset=int(payload.get("offset", 0)),
                scope=c.SCOPE,
            )
        else:
            result = c.compact_thread(keep=1, scope=c.SCOPE)
            result["pointers"] = c.memory_pointers(c.SCOPE)
    elif number == "12":
        result = {
            "usage": user_calls.get(ACTIVE_SCOPE.owner, [])[-30:],
            "searches": user_searches.get(ACTIVE_SCOPE.owner, [])[-20:],
            "context": context(),
            "backend": BACKEND,
        }
        if payload.get("probe_memories"):
            foreign = replace(c.SCOPE, user_id="__isolation_probe__")
            another_thread = replace(c.SCOPE, thread_id="__missing_thread__")
            result["memory_probes"] = {
                kind: {
                    "owned_count": len(
                        c.retrieve(
                            query or "trip preferences", kind=kind, scope=c.SCOPE
                        )
                    ),
                    "foreign_owner_count": len(
                        c.retrieve(
                            query or "trip preferences", kind=kind, scope=foreign
                        )
                    ),
                    "other_thread_count": len(
                        c.retrieve(
                            query or "trip preferences", kind=kind, scope=another_thread
                        )
                    ),
                }
                for kind in ["conversation", "workflow", "summary"]
            }
            probe_scope = replace(
                c.SCOPE, thread_id="failure_probe_" + c.uuid.uuid4().hex[:10]
            )
            outcome, finished = c.execute_step(
                {"action": "unpack", "memory_id": "__missing_memory__", "offset": 0},
                c.uuid.uuid4().hex,
                0,
                scope=probe_scope,
            )
            result["recorded_failure"] = c.workflow_memory(1, probe_scope)[0]
        if payload.get("exercise_threshold"):
            full = c.context_tokens(
                c.build_context(query or "Inspect trip memory.", "threshold", c.SCOPE)
            )
            minimal = c.context_tokens(
                c.build_context(
                    query or "Inspect trip memory.",
                    "threshold",
                    c.SCOPE,
                    workflow_limit=1,
                    conversation_limit=0,
                )
            )
            if full - minimal < 400:
                raise ValueError(
                    "Run a research turn first so there is actual context to compact."
                )
            original_limit = c.CONTEXT_LIMIT
            try:
                c.CONTEXT_LIMIT = max(minimal + 256, full - 64)
                selected, budget = c.bounded_context(
                    query or "Inspect trip memory.", "threshold", c.SCOPE
                )
                result["threshold_experiment"] = {"context": selected, "budget": budget}
            finally:
                c.CONTEXT_LIMIT = original_limit
    else:
        raise ValueError("Unknown Part.")
    return {**result, "calls": c.CALLS[start:]}
