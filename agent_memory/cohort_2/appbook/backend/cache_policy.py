"""Appbook semantic cache policy: paraphrase reuse without crossing opposite intents.

The from-scratch notebook uses a loose 0.3 cosine-distance threshold so it can show
both paraphrase reuse and a wrong hit for a different question. The appbook answers
whatever the learner types, so it replaces the runtime modules' threshold with a
tighter one here and, like the notebook's negation and number guard, places
contrasting terms in the cache identity. Measured with all-MiniLM-L6-v2: close
paraphrases sit at 0.01–0.10, while short-term/long-term and
advantages/disadvantages questions sit at 0.04–0.06, so distance alone cannot
separate them. 0.08 keeps most paraphrases and misses the rest, which is safe.
The generated *_core.py modules are left untouched.
"""
import json
import re

SCORE_THRESHOLD = 0.08

# Each form maps to a canonical term, so "advantage" and "advantages" share a namespace
# while "advantages" and "disadvantages" never do.
CONTRAST_TERMS = {
    **dict.fromkeys(["short"], "short"),
    **dict.fromkeys(["long"], "long"),
    **dict.fromkeys(["advantage", "advantages", "benefit", "benefits", "pro", "pros"], "advantage"),
    **dict.fromkeys(["disadvantage", "disadvantages", "drawback", "drawbacks", "downside",
                     "downsides", "con", "cons", "risk", "risks", "limitation", "limitations"], "disadvantage"),
    **dict.fromkeys(["best"], "best"),
    **dict.fromkeys(["worst"], "worst"),
    **dict.fromkeys(["increase", "increases", "raise", "raises"], "increase"),
    **dict.fromkeys(["decrease", "decreases", "reduce", "reduces", "lower", "lowers"], "decrease"),
    **dict.fromkeys(["read", "reads", "reading", "retrieve", "retrieves", "recall", "recalls"], "read"),
    **dict.fromkeys(["write", "writes", "writing", "store", "stores"], "write"),
    **dict.fromkeys(["forget", "forgets", "forgetting", "delete", "deletes", "evict", "evicts"], "forget"),
    **dict.fromkeys(["update", "updates", "correct", "corrects", "overwrite", "overwrites"], "update"),
    **dict.fromkeys(["before"], "before"),
    **dict.fromkeys(["after"], "after"),
}

MEMORY_CONCEPT = re.compile(
    r"\b(memor(y|ies)|cach(e|es|ed|ing)|context|retrieval|compaction|compression|"
    r"summar(y|ies|ization|isation)|vectors?|embeddings?|rerank\w*|rag|tokens?)\b"
)


def contrast_terms(question):
    words = re.findall(r"[a-z]+", question.lower())
    return {CONTRAST_TERMS[word] for word in words if word in CONTRAST_TERMS}


def memory_concept(question):
    """A stable agent-memory explanation, as opposed to a live travel request."""
    return bool(MEMORY_CONCEPT.search(question.lower()))


def apply(core):
    """Configure one generated runtime module's semantic cache; safe to call repeatedly."""
    if getattr(core, "_appbook_cache_policy", False):
        return core
    if not hasattr(core.semantic_cache, "_score_threshold"):
        raise RuntimeError("OracleSemanticCache no longer exposes _score_threshold; review cache_policy.py.")
    core.semantic_cache._score_threshold = SCORE_THRESHOLD
    notebook_namespace = core.cache_namespace

    def cache_namespace(question, *args, **kwargs):
        identity = json.loads(notebook_namespace(question, *args, **kwargs))
        identity["risk_signature"] = sorted(set(identity["risk_signature"]) | contrast_terms(question))
        return core.pretty(identity)

    core.cache_namespace = cache_namespace
    core._appbook_cache_policy = True
    return core
