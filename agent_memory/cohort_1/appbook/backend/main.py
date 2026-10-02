"""FastAPI appbook with live providers, scoped memory and visible context windows."""
import array
import json
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT.parent
config = ROOT / "data/oracle_v2.json"
if config.exists():
    for key, value in json.loads(config.read_text()).items():
        os.environ.setdefault(key, value)
os.environ["NOTEBOOK_USE_ENV_KEYS"] = "1"
os.environ.setdefault("TRAVEL_THREAD", "appbook_active_trip")
from . import runtime as rt
from . import interactions, tokenomics, workloads

TABLES = {
    "AM_STATE_V2": "Versioned trip session and typed profile bridge",
    "AM_EVENTS_V2": "Host events and entity provenance",
    "AM_BLOBS_V2": "Compressed source archives with checksums",
    "AM_SEMANTIC_CACHE_V2": "Scoped educational answer cache",
    "KNOWLEDGE_BASE": "Memorizz source passages and HNSW vectors",
    "CONVERSATION_MEMORY": "Memorizz original turns and compaction markers",
    "ENTITY_MEMORY": "Memorizz current preferences and source attribution",
    "WORKFLOW_MEMORY": "Memorizz agent decisions and execution outcomes",
    "SUMMARIES": "Memorizz summaries linked to original source turns",
}


def safe(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, array.array):
        return {"dimensions": len(value), "preview": list(value[:8])}
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=rt.warm, daemon=True).start()
    yield
    with rt.lock:
        if rt.core:
            if hasattr(rt.core, "provider"):
                rt.core.provider.close()
            rt.core.conn.close()
            rt.core.client.close()


app = FastAPI(title="Wayfinder · Memory aware travel appbook", lifespan=lifespan)


@app.middleware("http")
async def local_origin(request: Request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if origin and origin not in {
            f'http://{request.headers.get("host")}',
            f'https://{request.headers.get("host")}',
        }:
            return JSONResponse(
                {"detail": "Cross-origin writes are not accepted."}, status_code=403
            )
    return await call_next(request)


@app.exception_handler(Exception)
@app.exception_handler(ValueError)
@app.exception_handler(RuntimeError)
async def failed(request, exc):
    message = str(exc)
    for name in [
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "ORACLE_PASSWORD",
        "TYPESAFE_API_KEY",
    ]:
        if os.getenv(name):
            message = message.replace(os.environ[name], "[redacted]")
    message = re.sub(r"(sk-ant-|tvly-)[A-Za-z0-9_-]+", "[redacted]", message)
    return JSONResponse(
        {"detail": message[:350]},
        status_code=409 if isinstance(exc, (RuntimeError, ValueError)) else 500,
    )


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class Trip(BaseModel):
    request: str = Field(min_length=1, max_length=4000)
    new_thread: bool = False


class Conversation(BaseModel):
    keep_trip: bool = True


class UserProfile(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)


class UserSelection(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)


class Interaction(BaseModel):
    query: str = Field(default="", max_length=4000)
    memory_text: str = Field(default="", max_length=10000)
    fetch_web: bool = False
    cache_id: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,80}$")


class Experiment(BaseModel):
    agents: list[str] = Field(min_length=1, max_length=4)
    prompts: list[str] = Field(default_factory=list, max_length=30)
    mode: str = Field(default="custom", pattern=r"^(scenario|synthetic|custom)$")
    scenario: str = Field(default="research", pattern=r"^(research|preferences|context|cache)$")
    workload_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    turns: int = Field(default=4, ge=1, le=30)
    options: dict[str, bool] = Field(default_factory=dict)
    context_limit: int = Field(default=24000, ge=3000, le=100000)

    @field_validator("agents")
    @classmethod
    def agents_valid(cls, names):
        if len(set(names)) != len(names) or set(names) - set(tokenomics.LABELS):
            raise ValueError("Choose distinct supported agents.")
        return names

    @field_validator("prompts")
    @classmethod
    def prompts_valid(cls, prompts):
        if any(not p.strip() or len(p) > 4000 for p in prompts):
            raise ValueError("Each turn needs between 1 and 4,000 characters.")
        return [p.strip() for p in prompts]

    @field_validator("options")
    @classmethod
    def options_valid(cls, options):
        if set(options) - set(tokenomics.DEFAULT_OPTIONS):
            raise ValueError("Unknown experiment option.")
        return options


class Workload(BaseModel):
    mode: str = Field(default="scenario", pattern=r"^(scenario|synthetic)$")
    scenario: str = Field(default="research", pattern=r"^(research|preferences|context|cache)$")
    turns: int = Field(default=4, ge=1, le=30)
    focus: str = Field(default="", max_length=2000)


@app.post("/api/workloads")
def workload(payload: Workload):
    with rt.lock:
        return safe(workloads.create(payload.model_dump()))


@app.post("/api/compaction")
def compact():
    with rt.lock:
        return safe(interactions.compact_now())


@app.post("/api/profile/refresh")
def profile_refresh():
    with rt.lock:
        c = rt.require()
        turns = c.recent_events("conversation", limit=100, scope=c.SCOPE)
        for turn in turns:
            if turn["payload"]["role"] == "user":
                c.update_profile(c.extract_entities(turn["payload"]["content"]), turn["event_id"], c.SCOPE)
        return safe(rt.snapshot())


@app.get("/api/status")
def status():
    return {
        "ready": rt.ready,
        "error": rt.error,
        "oracle": "Connected" if rt.ready else "Warming",
        "responder": os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"),
        "memory_backend": rt.BACKEND,
        "live_search": "Tavily",
        "decision_available": bool(os.getenv("TYPESAFE_API_KEY")),
        "memorizz_version": getattr(rt.core, "MEMORIZZ_VERSION", None) if rt.ready else None,
    }


@app.get("/api/state")
def state():
    with rt.lock:
        return safe(rt.snapshot())


@app.get("/api/users")
def users():
    with rt.lock:
        return safe({"users": rt.users(), "active_user_id": rt.ACTIVE_SCOPE.user_id})


@app.post("/api/users")
def new_user(payload: UserProfile):
    with rt.lock:
        return safe(rt.create_user(payload.display_name))


@app.post("/api/users/select")
def select_user(payload: UserSelection):
    with rt.lock:
        return safe(rt.switch_user(payload.user_id))


@app.post("/api/chat")
def chat(payload: Chat):
    with rt.lock:
        return safe(rt.chat(payload.message))


@app.post("/api/trip")
def trip(payload: Trip):
    with rt.lock:
        return safe(rt.save_trip(payload.request, payload.new_thread))


@app.post("/api/conversations")
def conversation(payload: Conversation):
    with rt.lock:
        return safe(rt.new_conversation(payload.keep_trip))


@app.post("/api/embeddings")
def embedding_space(payload: Interaction):
    with rt.lock:
        return safe(interactions.embeddings_space(payload.query, payload.memory_text))


@app.post("/api/compare/{technique}")
def comparison(technique: str, payload: Interaction):
    if not payload.query.strip():
        raise ValueError("Enter a query.")
    with rt.lock:
        if technique == "retrieval":
            result = interactions.retrieval(payload.query, payload.fetch_web)
        elif technique == "reranking":
            result = interactions.reranking(payload.query, payload.fetch_web)
        elif technique == "semantic-cache":
            result = interactions.semantic_comparison(payload.query, payload.cache_id)
        elif technique == "compaction":
            result = interactions.compaction_comparison(payload.query)
        else:
            raise HTTPException(404, "Unknown comparison.")
        return safe(result)


@app.post("/api/tokenomics")
def experiment(payload: Experiment):
    with rt.lock:
        return safe(tokenomics.start(payload.model_dump()))


@app.get("/api/tokenomics/{job_id}")
def experiment_progress(job_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(404, "Unknown experiment.")
    return safe(tokenomics.get(job_id))


@app.post("/api/tokenomics/{job_id}/cancel")
def experiment_cancel(job_id: str):
    return safe(tokenomics.cancel(job_id))


@app.get("/api/context")
def context():
    with rt.lock:
        return safe(rt.context())


@app.post("/api/labs/{number}")
def lab(number: str, payload: dict):
    if number not in {str(n) for n in range(13)}:
        raise HTTPException(404, "Unknown Part.")
    if len(json.dumps(payload)) > 6000:
        raise HTTPException(422, "Experiment input too long.")
    with rt.lock:
        return safe(rt.lab(number, payload))


def available_tables():
    c = rt.require()
    existing = {r["table_name"] for r in c.rows("SELECT table_name FROM user_tables")}
    options = {**TABLES, "AM_MEMORY_V2": "OracleVS source passages and HNSW vectors"}
    return {name: label for name, label in options.items() if name in existing}


def table_scope(name):
    """The explorer follows the selected traveler, including a truly empty user."""
    if name.startswith("AM_") and name not in {"AM_MEMORY_V2", "AM_SEMANTIC_CACHE_V2"}:
        return "owner_id = :owner"
    if name == "AM_MEMORY_V2":
        return "JSON_VALUE(metadata, '$.owner' NULL ON ERROR) = :owner"
    if name == "AM_SEMANTIC_CACHE_V2":
        return "JSON_VALUE(metadata, '$.llm_string_hash') IN (SELECT namespace_hash FROM AM_CACHE_SCOPES_V2 WHERE owner_id=:owner)"
    return "user_id = :owner"


@app.get("/api/tables")
def tables():
    with rt.lock:
        c = rt.require()
        return [
            {
                "name": name,
                "description": label,
                "count": int(c.rows(f"SELECT COUNT(*) AS n FROM {name} WHERE {table_scope(name)}",
                                    {"owner": rt.ACTIVE_SCOPE.owner})[0]["n"]),
            }
            for name, label in available_tables().items()
        ]


@app.get("/api/tables/{name}")
def table(name: str):
    with rt.lock:
        if name not in available_tables():
            raise HTTPException(404, "Unknown course table.")
        result = rt.require().rows(f"SELECT * FROM {name} WHERE {table_scope(name)} FETCH FIRST 40 ROWS ONLY",
                                   {"owner": rt.ACTIVE_SCOPE.owner})
        return {"name": name, "rows": safe(result), "limit": 40}


@app.get("/api/live-validation")
def validation():
    path = COHORT / (
        "memorizz_live_validation_report.json"
        if rt.BACKEND == "memorizz"
        else "live_validation_report.json"
    )
    return json.loads(path.read_text()) if path.exists() else {"status": "not_run"}


app.mount("/assets", StaticFiles(directory=str(COHORT / "data")))
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True))
