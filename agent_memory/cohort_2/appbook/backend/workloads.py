"""Resolve shared experiment inputs; generated turns never fabricate measurements."""
import json
import time
import uuid

from . import runtime as rt

WORKLOADS = {}

SCENARIOS = {
    "research": [
        "Recall my current destination, dates and budget. Research flight options for this trip and cite supplier sources.",
        "Using the same trip, research hotels that fit my saved preferences. Cite the sources and state which terms need checking.",
        "Research airport transport for that destination and compare public transport with a taxi.",
        "Compare the baggage and change policies of the flight suppliers you found.",
        "Which hotel cancellation terms matter for my saved refundable-room preference?",
        "Combine the researched options into a shortlist within my current budget; separate unknown prices.",
    ],
    "preferences": [
        "For this experiment, remember that I prefer a window seat and a quiet hotel room.",
        "What destination am I travelling to, and which seat and room preferences do you remember?",
        "Correct my seat preference to aisle. Keep the quiet-room preference.",
        "Research hotels for the active destination using my remembered room preference.",
        "Remember that I need one checked bag and a refundable hotel.",
        "Summarize my current preferences, including the seat correction, and how they affect the trip.",
    ],
    "context": [
        "Research the active trip and explain the flight options, supplier policies and sources in detail.",
        "Compare hotel locations at the destination using my current budget and preferences.",
        "Compare airport transfers and explain their practical trade-offs.",
        "Create a detailed arrival-day itinerary using the information already researched.",
        "Explain how baggage, cancellation and transport choices affect the plan. Refer to the sources already found.",
        "Recall the decisions we have made so far, and identify unanswered questions before proceeding.",
    ],
    "cache": [
        "Explain how semantic cache helps agent memory in two sentences.",
        "Explain how semantic cache helps agent memory in two sentences.",
        "In two sentences, how does semantic caching help an agent reuse memory?",
        "Explain how prompt caching helps context engineering in two sentences.",
        "Explain how prompt caching helps context engineering in two sentences.",
        "Explain how embedding reuse helps agent memory in two sentences.",
    ],
}


def preset(scenario, turns):
    if scenario not in SCENARIOS:
        raise ValueError("Choose a supported travel scenario.")
    base = SCENARIOS[scenario]
    if scenario == "cache":
        # Intentional repetition is the workload under test, not a fallback.
        return [base[i % len(base)] for i in range(turns)]
    sequence = base[:turns]
    topics = ["flight choices", "hotel terms", "airport transport", "budget allocation", "arrival-day timing", "supplier policy uncertainties"]
    for i in range(len(sequence), turns):
        topic = topics[(i - len(base)) % len(topics)]
        sequence.append(
            f"Follow-up {i + 1}: revisit {topic} using the latest trip, corrected preferences and earlier outcomes. "
            "Identify one remaining decision, explain its trade-off and avoid repeating completed research."
        )
    return sequence


def preview(payload):
    mode, scenario, turns = payload["mode"], payload["scenario"], payload["turns"]
    if mode == "scenario":
        return {"mode": mode, "scenario": scenario, "prompts": preset(scenario, turns), "generation": None}
    if mode != "synthetic":
        raise ValueError("Choose a scenario or synthetic generation.")
    c = rt.require()
    session = c.read_state(scope=c.SCOPE)
    initial = {"trip": session["state"].get("trip", {}) if session else {},
               "profile": {k: v["value"] for k, v in c.load_profile(c.SCOPE).items()}}
    before, started = len(c.CALLS), time.perf_counter()
    generated = c.llm_json(
        c.pretty({"scenario": scenario, "turns": turns, "starting_facts": initial,
                  "focus": payload.get("focus", ""), "example_sequence": SCENARIOS[scenario]}),
        system=("Generate an ordered synthetic traveler conversation for a live agent experiment. "
                "Return JSON with just a prompts array of exactly the requested number of nonempty user messages. "
                "Vary the turns; include follow-ups, corrections and recall of previous decisions. "
                "Use the supplied active trip; if missing, first ask for the missing information. "
                "Do not invent prices, availability, supplier facts, booking outcomes or assistant answers. "
                "Any new preferences must be explicitly stated in a generated user message and remain experiment-only. "
                "Repeat requests only if the scenario explicitly tests cache reuse. "
                "Each message must be no longer than 4000 characters."),
        purpose="shared experiment workload generation",
    )
    prompts = generated.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != turns or any(
        not isinstance(p, str) or not p.strip() or len(p) > 4000 for p in prompts
    ):
        raise ValueError("Generated sequence was invalid; no experiment was started.")
    calls = c.CALLS[before:]
    costs = [call["estimated_usd"] for call in calls]
    generation = {"seconds": time.perf_counter() - started, "calls": calls,
                  "estimated_usd": sum(costs) if all(v is not None for v in costs) else None,
                  "model": c.LLM_MODEL_ID, "charged_once": True}
    return {"mode": mode, "scenario": scenario, "prompts": [p.strip() for p in prompts], "generation": generation}


def create(payload):
    result = {"id": uuid.uuid4().hex, **preview(payload)}
    result["owner"] = rt.require().SCOPE.owner
    result["thread"] = rt.require().SCOPE.thread_id
    WORKLOADS[result["id"]] = result
    return result


def resolve(payload):
    if payload.get("workload_id"):
        result = WORKLOADS.get(payload["workload_id"])
        c = rt.require()
        if not result or result["owner"] != c.SCOPE.owner or result["thread"] != c.SCOPE.thread_id:
            raise ValueError("Preview this workload again for the current conversation.")
        if len(result["prompts"]) != payload["turns"]:
            raise ValueError("Turn count changed; preview the workload again.")
        return result
    if payload.get("mode") == "scenario":
        return {"mode": "scenario", "scenario": payload["scenario"],
                "prompts": preset(payload["scenario"], payload["turns"]), "generation": None}
    if payload.get("mode") == "synthetic":
        raise ValueError("Generate and review the synthetic turns before running them.")
    prompts = payload["prompts"]
    if len(prompts) != payload["turns"]:
        raise ValueError("Custom sequences need one request for every turn; requests are not automatically repeated.")
    return {"mode": "custom", "scenario": None, "prompts": prompts, "generation": None}
