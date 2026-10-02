"""Generate SVG diagram sources; rasterize with Chromium for notebook compatibility."""
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FLOW = [
    ("Traveler request", "User supplies trip details and explicit preferences"),
    ("Load current memory", "Session · entity profile · conversation · workflow outcomes"),
    ("Construct the context", "Stable cached prefix → changing memory suffix"),
    ("Count tokens and compact", "Host threshold + agent-requested compaction"),
    ("Model chooses an action", "Raw Anthropic · cache read/write usage is visible"),
    ("Retrieve or search", "Oracle HNSW → reranker · Tavily live web search"),
    ("Offload and record", "Full tool payload → compressed Oracle object · success/failure event"),
    ("Read details just in time", "ID + description → unpack summary or original result"),
    ("Grounded research shortlist", "Actual source links · no invented fares or reservations"),
]
TIERS = [
    ("Traveler and channel", [("Traveler", "Own request and corrections"), ("Appbook", "Traveler profiles + context view")]),
    ("Application control", [("Scope and session", "Host-owned identity and version"), ("Context builder", "Pinned state + bounded memories"), ("Agent loop", "Validated actions and budgets"), ("Semantic cache", "Stable answers only")]),
    ("Models", [("MiniLM encoder", "384-dimensional semantic recall"), ("Cross-encoder", "Question/passage relevance"), ("Claude Opus 5.5", "Raw Anthropic + prompt cache")]),
    ("Trusted capabilities", [("HNSW retrieval", "Owner + kind + vector identity"), ("Tavily search", "Live flights, hotels, transport"), ("Offload / unpack", "Discoverable IDs + checksums"), ("Compaction", "LLM summary + source archive")]),
    ("Oracle durable memory", [("OracleVS", "AM_MEMORY_V2 + HNSW"), ("State", "Trip and current entities"), ("Events", "Conversation + step outcomes"), ("Compressed objects", "Tool payloads and summaries")]),
    ("Inspection", [("Context window", "Prefix and suffix separately"), ("Provider counters", "Cache reads, writes and costs"), ("Oracle explorer", "Actual persisted rows")]),
]


def svg_document(width, height, body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#07100e"/><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10" fill="none" stroke="#b5fa52"/></marker></defs><style>text{{font-family:Arial,sans-serif;fill:#e5eee9}}.title{{font-size:23px;font-weight:bold}}.note{{font-size:16px;fill:#a8beb2}}.tier{{font-size:18px;font-weight:bold;fill:#b5fa52}}</style>{body}</svg>'


def build():
    body = ['<text x="50" y="55" class="title">Memory-aware travel research · one visible loop</text>']
    for i, (title, note) in enumerate(FLOW):
        y = 90 + i * 115
        if i:
            body.append(f'<path d="M560,{y-25} V{y-5}" stroke="#b5fa52" stroke-width="3" marker-end="url(#arrow)"/>')
        body.append(f'<rect x="65" y="{y}" width="990" height="90" rx="12" fill="#10201a" stroke="#31503f"/>')
        body.append(f'<text x="90" y="{y+33}" class="title">{i+1}. {html.escape(title)}</text>')
        body.append(f'<text x="90" y="{y+64}" class="note">{html.escape(note)}</text>')
    (DATA / "travel_assistant_flow.svg").write_text(svg_document(1120, 1170, "".join(body)))

    import copy
    editions = {
        "custom": ("Custom memory agent · manual loop", copy.deepcopy(TIERS)),
        "memorizz": ("Memorizz memory components · manual loop", copy.deepcopy(TIERS)),
        "decisions": ("Custom agent + typed decision models", copy.deepcopy(TIERS)),
    }
    editions["memorizz"][1][1][1][2] = ("Manual agent loop", "Explicit context construction")
    editions["memorizz"][1][4] = ("Oracle via published Memorizz", [("KnowledgeBase", "Native HNSW knowledge vectors"), ("Trip + entities", "Host state + EntityMemory"), ("Turns + workflows", "Native memory units"), ("Summaries + originals", "Atomic links + host archives")])
    editions["memorizz"][1][3][1][0] = ("HNSW retrieval", "MemoryManager + OracleProvider")
    editions["decisions"][1][2][1].append(("Jev or CLEF", "Typed gates and routing"))
    editions["decisions"][1][2][1][1] = ("Decision reranker", "Jev / CLEF passage selection")
    for edition, (title, tiers) in editions.items():
        body = [f'<text x="35" y="48" class="title">{html.escape(title)}</text>']
        for i, (tier, nodes) in enumerate(tiers):
            y = 75 + i * 180
            body.append(f'<rect x="20" y="{y}" width="1080" height="152" rx="14" fill="#0c1b15" stroke="#274334"/>')
            body.append(f'<text x="40" y="{y+30}" class="tier">{html.escape(tier)}</text>')
            for j, (name, tech) in enumerate(nodes):
                x = 38 + j * 267
                body.append(f'<rect x="{x}" y="{y+52}" width="248" height="76" rx="10" fill="#12251c" stroke="#467051"/>')
                body.append(f'<text x="{x+13}" y="{y+81}" font-size="18" font-weight="bold">{html.escape(name)}</text>')
                body.append(f'<text x="{x+13}" y="{y+108}" font-size="13" fill="#a8beb2">{html.escape(tech)}</text>')
            if i < len(tiers)-1:
                body.append(f'<path d="M560,{y+154} V{y+176}" stroke="#b5fa52" stroke-width="3" marker-end="url(#arrow)"/>')
        (DATA / f"travel_architecture_{edition}.svg").write_text(svg_document(1120, 1180, "".join(body)))

    from playwright.sync_api import sync_playwright
    executable = "/Users/richmondalake/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=executable)
        for name in ["travel_assistant_flow", "travel_architecture_custom", "travel_architecture_memorizz", "travel_architecture_decisions"]:
            page = browser.new_page(viewport={"width":1120, "height":1180}, device_scale_factor=1.5)
            page.goto((DATA / (name+".svg")).as_uri())
            page.locator("svg").screenshot(path=str(DATA / (name+".png")))
            page.close()
        browser.close()
    print("Generated standalone SVG and embedded-compatible PNG diagrams.")


if __name__ == "__main__":
    build()
