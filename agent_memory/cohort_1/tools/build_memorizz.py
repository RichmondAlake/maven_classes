"""Build the framework edition while preserving Parts 0–12 and live examples."""
import ast
import copy
import hashlib
from pathlib import Path
import black
import nbformat
from build_notebooks import ROOT
from cell_explanations import ensure_explanations
from notebook_setup import SETUP_TAG, add_dependency_setup

SOURCE = ROOT / 'tools/memorizz_components.py'
text = SOURCE.read_text().replace('return DIM\n', 'return DIMENSIONS\n').replace('"dimensions": DIM,', '"dimensions": DIMENSIONS,')
tree = ast.parse(text)
functions = {node.name: ast.get_source_segment(text, node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
base = nbformat.read(ROOT / 'implementing_memory_aware_agents.ipynb', 4)
cells = []


def add_code(source, explanation, tag='runtime'):
    cells.append(nbformat.v4.new_markdown_cell(explanation))
    cells.append(nbformat.v4.new_code_cell(black.format_str(source, mode=black.Mode(line_length=88)), metadata={'tags': [tag]}))


for original in base.cells:
    if SETUP_TAG in original.metadata.get('tags', []):
        continue
    cell = copy.deepcopy(original)
    if cell.cell_type == 'markdown':
        cell.source = cell.source.replace('Implementing Memory Aware Agents\n', 'Implementing Memory Aware Agents · Memorizz edition\n', 1)
        cell.source = cell.source.replace('**OracleVS** owns the vector table and batch insertion.', '**Memorizz OracleProvider** owns the native memory tables. KnowledgeBase chunks sources and MemoryManager retrieves scoped candidates.').replace('through OracleVS.', 'through Memorizz KnowledgeBase.')
        cell.source = cell.source.replace('OracleVS ingestion', 'Memorizz KnowledgeBase ingestion')
        cell.source = cell.source.replace("OracleVS's approximate search", "Memorizz's explicitly enabled approximate search")
        cell.source = cell.source.replace('The OracleVS adapter', 'The Memorizz embedding adapter')
        cell.source = cell.source.replace('OracleVS **', 'Memorizz **')
        if '### Reference architecture' in cell.source:
            cell.source = cell.source.replace('attachment:travel_architecture_custom.png', 'attachment:travel_architecture_memorizz.png')
            cell.source = cell.source.replace('This notebook builds the **custom memory agent**: an explicit loop with\nOracleVS/HNSW retrieval, current trip and entity facts, conversation, workflow\noutcomes and compressed archives.', 'This notebook keeps the **manual agent loop** visible while using the published Memorizz package for KnowledgeBase, MemoryManager, ConversationMemoryUnit, EntityMemory, Workflow and atomic summary links. Versioned trip state and compressed archives remain explicit host components. The Tokenomics **Memorizz MemAgent** lane uses the framework’s native loop, ContextPolicy, RetrievalPolicy and tool-result offloading; that is a separate implementation from this notebook’s manual loop.')
            cell.attachments = {}
            from build_notebooks import attach_diagrams
            attach_diagrams([cell], ROOT / 'data')
        if '### Part 2.2' in cell.source:
            cell.source = '### Part 2.2 · Enable native HNSW retrieval\n\nMemorizz creates HNSW indexes under its lazy index policy. The published package supports **vector_search_mode="approximate"** to request approximate retrieval explicitly. Exact search remains the library default for existing applications. The next read creates the knowledge index; Part 2 checks the actual plan. A missing vector-memory pool must be fixed before continuing.'
        cells.append(cell)
        continue

    names = {node.name for node in ast.parse(cell.source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    if 'memory_store = OracleVS(' in cell.source:
        cells.append(nbformat.v4.new_markdown_cell('### Use the published Memorizz package\n\nInstall `requirements-memorizz.txt` in this notebook kernel. It pins the tested PyPI release. **importlib.metadata.version** reports the installed version; no checkout path is added to Python imports. Generation remains the raw Anthropic client. Memorizz supplies memory components and its Oracle provider.'))
        cell.source = '''from importlib.metadata import distribution, version
from pathlib import Path
import json

import memorizz
from langchain_oracledb.vectorstores import DistanceStrategy
from memorizz.embeddings import BaseEmbeddingProvider, set_global_embedding_manager
from memorizz.enums.memory_type import MemoryType
from memorizz.memory_provider.oracle import OracleConfig, OracleProvider
from memorizz.long_term.semantic.knowledge_base import KnowledgeBase
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.long_term.episodic.conversational_memory_unit import ConversationMemoryUnit
from memorizz.long_term.procedural.workflow import Workflow, WorkflowOutcome
from memorizz.memagent.managers.memory_manager import MemoryManager

MEMORIZZ_VERSION = version("memorizz")
installed = distribution("memorizz")
origin = json.loads(installed.read_text("direct_url.json") or "{}")

# An editable checkout would silently use local source instead of the pip release.
if origin.get("dir_info", {}).get("editable"):
    raise RuntimeError("Install the published Memorizz wheel in this kernel.")

print("Memorizz package version:", MEMORIZZ_VERSION)
print("Installed package:", Path(memorizz.__file__).parent)'''
        cells.append(cell)
        for name in ['memory_namespace', 'token_chunks', 'LocalTravelEmbeddings', 'retrieve_native_memory']:
            add_code(functions[name], f'**{name}**: {ast.get_docstring(ast.parse(functions[name]).body[0])}\n\nKnowledge chunks fit the same encoder used by the baseline. Longer structured workflow records average normalized chunk vectors, while their complete source remains durable. The embedding identity stays in every namespace.')
        add_code('''embedding_provider = LocalTravelEmbeddings()
set_global_embedding_manager(embedding_provider)
provider = OracleProvider(OracleConfig(
    user=DB_USER,
    password=DB_PASSWORD,
    dsn=DB_DSN,
    in_database_embedding=False,
    embedding_provider=embedding_provider,
    index_policy="lazy",
    vector_search_mode="approximate",
    pool_min=1,
    pool_max=3,
))
assert provider.preflight()["ok"]
manager = MemoryManager(provider)
knowledge = KnowledgeBase(provider)
entity_memory = EntityMemory(provider)''', 'Create **OracleProvider(config)** with the same dedicated schema and local embedding provider. **MemoryManager**, **KnowledgeBase** and **EntityMemory** use this provider. The application still owns versioned session state, scoped archives and validation. Native conversation and workflow records replace their hand-built equivalents.')
        continue
    if 'create_index(' in cell.source:
        cell.source = '# Lazy index creation occurs through a real provider retrieval.\nmanager.retrieve_relevant_memories(\n    "travel policy", MemoryType.KNOWLEDGE_BASE, SCOPE.owner,\n    user_id=SCOPE.owner, namespace=memory_namespace("policy"), limit=1,\n)'
    elif 'append_event' in names:
        cell.source += '\n\nhost_append_event = append_event\n'
        cells.append(cell)
        add_code(functions['append_event'], '**append_event(kind, payload, status, run_id, step, scope)** writes ConversationMemoryUnit for actual turns and Workflow for each real execution outcome. Entity audit events retain the explicit host ledger. A workflow record describes one iteration, not an invented reusable procedure.')
        continue
    elif 'recent_events' in names:
        cell.source = ast.get_source_segment(cell.source, next(n for n in ast.parse(cell.source).body if isinstance(n, ast.FunctionDef) and n.name == 'recent_events')) + '\n\nhost_recent_events = recent_events'
        cells.append(cell)
        add_code(functions['recent_events'], '**recent_events(kind, limit, active_only, scope)** reads native conversation history. Its relational workflow read orders steps by time and applies the owner and thread memory ID before the limit. This avoids the current convenience wrapper ignoring a dictionary filter.')
        add_code(functions['decode_entity_value'], '**decode_entity_value(value)** accepts the bridge’s JSON encoding and native MemAgent string attributes. A plain seat value such as `window` remains a string; encoded booleans retain their type. This lets the manual loop and native framework share entity memory without assuming identical serialization.')
        add_code(functions['load_profile'], '**load_profile(scope)** reads canonical facts from Memorizz EntityMemory. Values pass through the adapter above, and each retains its original turn ID. Current entities stay outside provider-cached prompt instructions.')
        continue
    elif 'update_profile' in names:
        start = cell.source.index('    source = rows(')
        end = cell.source.index('\n    if not source:', start)
        cell.source = cell.source[:start] + '    source = [turn for turn in recent_events("conversation", limit=100, scope=scope)\n              if turn["event_id"] == source_turn_id]\n' + cell.source[end:]
        cell.source = cell.source.replace('    return state', '    sync_entity_profile(scope)\n    return state')
        add_code(functions['sync_entity_profile'], '**sync_entity_profile(scope)** publishes host-validated current preferences through EntityMemory.upsert_entity. The versioned host state serializes updates; entity attributes retain their source turn for inspection. This is a two-write teaching bridge, not a claim of distributed transaction atomicity.')
    elif 'load_blob' in names:
        cell.source = ast.get_source_segment(cell.source, next(n for n in ast.parse(cell.source).body if isinstance(n, ast.FunctionDef) and n.name == 'load_blob'))
        cells.append(cell)
        add_code(functions['memory_pointers'], '**memory_pointers(scope, limit)** exposes only published summaries. Native source linking and compaction markers are atomic in Memorizz; the archive write precedes that transaction. Failed linking removes the unpublished archive. Search archives remain available immediately.')
        continue
    elif names & functions.keys():
        cell.source = '\n\n'.join(functions.get(node.name, ast.get_source_segment(cell.source, node)) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else ast.get_source_segment(cell.source, node) for node in ast.parse(cell.source).body)
    cell.source = cell.source.replace("table_name = 'AM_MEMORY_V2'", "table_name = 'KNOWLEDGE_BASE'")
    if cell.source.startswith('conn.close()\nclient.close()'):
        cell.source = 'provider.close()\n' + cell.source
    cell.source = black.format_str(cell.source, mode=black.Mode(line_length=88))
    cells.append(cell)

cells.insert(1, nbformat.v4.new_markdown_cell('### What changes in this edition?\n\nThe educational Parts and live travel use case are preserved. Memorizz supplies KnowledgeBase ingestion, MemoryManager retrieval, ConversationMemoryUnit, EntityMemory, Workflow and atomic summary/source links. OracleSemanticCache remains the requested langchain_oracledb abstraction around raw Anthropic. Versioned host state and compressed archives stay explicit so their context-engineering behavior is visible. Install `requirements-memorizz.txt`, which uses the published Memorizz 0.13.0 package. This notebook retains the manual loop so every context and memory step is visible. The appbook Tokenomics comparison also runs native MemAgent as a separate implementation.'))
cells = ensure_explanations(cells)
cells = add_dependency_setup(cells, 'requirements-memorizz.txt')
nb = nbformat.v4.new_notebook(cells=cells, metadata=copy.deepcopy(base.metadata))
nb.metadata.memorizz_package = 'memorizz==0.13.0'
for index, cell in enumerate(cells):
    cell.id = hashlib.sha256(f'memorizz:{index}:{cell.source}'.encode()).hexdigest()[:12]
nbformat.validate(nb)
nbformat.write(nb, ROOT / 'implementing_memory_aware_agents_memorizz.ipynb')
print('Memorizz edition:', len(cells), 'cells')
