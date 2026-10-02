"""Keep the travel lesson order while integrating typed decision models."""
import ast
import copy
import hashlib
import black
import nbformat
from build_notebooks import ROOT
from cell_explanations import ensure_explanations
from notebook_setup import SETUP_TAG, add_dependency_setup

text = (ROOT / 'tools/decision_components.py').read_text()
functions = {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
base = nbformat.read(ROOT / 'implementing_memory_aware_agents.ipynb', 4)
cells = []


def add(source, explanation, tag='runtime'):
    cells.append(nbformat.v4.new_markdown_cell(explanation))
    cells.append(nbformat.v4.new_code_cell(black.format_str(source, mode=black.Mode(line_length=88)), metadata={'tags': [tag]}))


for original in base.cells:
    if SETUP_TAG in original.metadata.get('tags', []):
        continue
    cell = copy.deepcopy(original)
    if cell.cell_type == 'markdown':
        if '### Reference architecture' in cell.source:
            cell.source = cell.source.replace('attachment:travel_architecture_custom.png', 'attachment:travel_architecture_decisions.png')
            cell.source = cell.source.replace('This notebook builds the **custom memory agent**: an explicit loop with', 'This notebook builds the **custom agent with typed decisions**: Jev or open CLEF adds entity-presence gates, capability routing, passage selection and summary quality checks to the explicit loop with')
            cell.attachments = {}
            from build_notebooks import attach_diagrams
            attach_diagrams([cell], ROOT / 'data')

        cell.source = cell.source.replace('Implementing Memory Aware Agents\n', 'Implementing Memory Aware Agents · Decision models\n', 1)
        cells.append(cell)
        continue
    names = {n.name for n in ast.parse(cell.source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if 'def rerank(' in cell.source:
        cells.append(cell)
        add('cross_encoder_rerank = rerank\nimport requests\n\nDECISION_CALLS = []\nENTITY_DECISIONS = []\nSELECTION_DECISIONS = []\nSUMMARY_DECISIONS = []', '### Part 3.1 · Decisions before generation\n\nInspired by your **01_reranking**, **02_summary_quality_gate** and **04_toolbox_skillbox_selection** notebooks, keep the candidate set fixed and compare readers. **Noul** reports a probability of yes, **Score** a weighted rubric level, and **Choice** an eligible option with probabilities. These models make typed decisions; Claude still writes answers and extracts entities.\n\n[Jev API](https://docs.typesafe.ai/api) · [Cloudflare CLEF release](https://huggingface.co/Cloudflare/clef)\n\nThe next cell saves the original cross-encoder for comparison and creates independent observation ledgers.')
        for name in ['validate_decisions', 'JevDecisions', 'ClefDecisions']:
            add(functions[name], '**' + name + '**: ' + ast.get_docstring(ast.parse(functions[name]).body[0]) + '\n\nBoth adapters use the same typed question contract. Hosted requests keep private keys out of logs. The open release is 27B parameters; its model card tests a CUDA environment with torch 2.11 and transformers 5.10.2 on an H200. Run it in a separate GPU kernel or compatible deployment; do not install that dependency set over this CPU teaching environment.')
        add('''jev_client = JevDecisions(request_secret("TYPESAFE_API_KEY"))
clef_release = os.getenv("CLEF_RELEASE_DIR")
clef_client = ClefDecisions(clef_release) if clef_release else None
backend = os.getenv("DECISION_BACKEND", "jev")
decision_client = {"jev": jev_client, "clef": clef_client}.get(backend)
if decision_client is None:
    raise ValueError("Selected decision backend is unavailable; configure CLEF_RELEASE_DIR on a suitable GPU.")
print("Active decision backend:", backend)
print("CLEF comparison:", "ready" if clef_client else "requires the downloaded release and CUDA GPU")''', 'Enter a hidden Jev key. **DECISION_BACKEND** selects the typed decision provider; it never disables Claude. To run the open weights, first download the reviewed Cloudflare/clef release into **CLEF_RELEASE_DIR** on a CUDA GPU with the model-card dependencies. This Mac validation executes Jev; it cannot validate the 27B CUDA model. An unavailable CLEF selection raises a clear error.')
        add(functions['decision_rerank'], '**decision_rerank(question, candidates, keep, decider)** scores each actual retrieved passage against an ordered relevance rubric. It preserves source metadata and before_rank. Similarity narrows memory; decision reranking chooses which memories deserve context space. No reranker can recover a missing candidate.')
        add('''def rerank(question, candidates, keep=3):
    return decision_rerank(question, candidates, keep=keep)''', 'Replace the agent’s reranking function with the selected typed decision reader. Subsequent RAG and recall calls use this definition. The original cross-encoder remains available for the comparison.')
        continue
    if 'comparison = pd.DataFrame' in cell.source:
        cells.append(cell)
        add('''reader_results = []
readers = {"cross_encoder": cross_encoder_rerank,
           "jev_closed": lambda q, c, keep: decision_rerank(q, c, keep, jev_client)}
if clef_client:
    readers["clef_open"] = lambda q, c, keep: decision_rerank(q, c, keep, clef_client)
for name, reader in readers.items():
    for rank, item in enumerate(reader(POLICY_QUERY, candidates, keep=len(candidates)), 1):
        reader_results.append({"reader": name, "before_rank": item["before_rank"],
                               "after_rank": rank, "score": item["rerank_score"],
                               "source": item["metadata"].get("url")})
display(pd.DataFrame(reader_results))''', 'Compare the same real candidate pool with the cross-encoder, hosted Jev and open CLEF when its GPU adapter is configured. Scores have different scales; compare order and source relevance, not raw score magnitude. A missing GPU produces no fabricated CLEF row.', 'example')
        continue
    if 'def extract_entities(' in cell.source:
        add(functions['entity_present'], '**entity_present(text, decider, threshold)** first asks whether the user explicitly states an identity fact such as their name, or a durable preference. Above the threshold, raw Claude extracts typed facts. This reduces unnecessary extraction calls but may miss useful entities; inspect the probability and validate on representative requests.')
    if 'entities = extract_entities(PREFERENCE_TEXT)' in cell.source:
        cell.source = cell.source.replace('entities = extract_entities(PREFERENCE_TEXT)', 'entities = extract_entities(PREFERENCE_TEXT) if entity_present(PREFERENCE_TEXT) else {"preferences": {}}')
    if 'def workflow_memory(' in cell.source:
        for name in ['eligible_catalog', 'select_capabilities', 'decision_route']:
            add(functions[name], '**' + name + '**: ' + ast.get_docstring(ast.parse(functions[name]).body[0]) + '\n\nHost eligibility precedes model choice. Toolbox entries describe executable reads; skillbox entries are authored reusable procedures. Include **none** and a confidence floor so a model can abstain. The selection enters the dynamic context suffix, after the stable cache breakpoint. Every routed tool outcome is still recorded as workflow memory.')
    if 'def compact_thread(' in cell.source:
        add(functions['summary_quality'], '**summary_quality(summary, turns, decider, support_floor, coverage_floor)** separately evaluates factual support and important-source coverage before any turn is marked compacted. A rejection preserves source turns. Thresholds are teaching defaults, not calibrated guarantees. Current session and preferences are supplied explicitly; archived history cannot override them.')
    if 'def agent_turn(' in cell.source:
        cell.source = cell.source.replace('    inspections = LAST_AGENT_RUN["inspections"]', '    inspections = LAST_AGENT_RUN["inspections"]\n    selected_procedure = None').replace('            usage_start = len(CALLS)', '            context["selected_procedure"] = selected_procedure\n            usage_start = len(CALLS)').replace('            decision_usage = CALLS[-1]', '            selected_procedure = context.get("decision_selection", selected_procedure)\n            decision_usage = CALLS[-1]')
    if 'agent_result = agent_turn(TRAVEL_REQUEST)' in cell.source:
        cell.source = cell.source.replace('agent_turn(TRAVEL_REQUEST)', 'agent_turn(TRAVEL_REQUEST, entity_gate=entity_present,\n                          quality_gate=summary_quality, route_decider=decision_route)')
    if 'compaction = compact_thread(keep=1)' in cell.source:
        cell.source = cell.source.replace('compact_thread(keep=1)', 'compact_thread(keep=1, quality_gate=summary_quality)')
    cell.source = black.format_str(cell.source, mode=black.Mode(line_length=88))
    cells.append(cell)

cells.insert(1, nbformat.v4.new_markdown_cell('### Closed and open decision models\n\nThis separate edition preserves Parts 0–12 and adds explicit typed decision hooks at their point of use. Jev runs as a hosted closed model. Cloudflare CLEF supplies open weights with the same question interface. Raw Anthropic remains mandatory for generation and entity extraction; no generation-off path is introduced. Live travel evidence still comes from Tavily and HNSW retrieval from Oracle.'))
cells = ensure_explanations(cells)
cells = add_dependency_setup(cells)
nb = nbformat.v4.new_notebook(cells=cells, metadata=copy.deepcopy(base.metadata))
for index, cell in enumerate(cells):
    cell.id = hashlib.sha256(f'decisions:{index}:{cell.source}'.encode()).hexdigest()[:12]
nbformat.validate(nb)
nbformat.write(nb, ROOT / 'implementing_memory_aware_agents_decision_models.ipynb')
print('Decision edition:', len(cells), 'cells')
