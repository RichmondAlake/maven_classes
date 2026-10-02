"""Extract application implementations without publishing lesson prose or code UI."""
import ast
from pathlib import Path
import nbformat

APP = Path(__file__).resolve().parent
COHORT = APP.parent


def build():
    for source, module in [
        ("implementing_memory_aware_agents.ipynb", "notebook_core"),
        ("implementing_memory_aware_agents_memorizz.ipynb", "memorizz_core"),
        ("implementing_memory_aware_agents_decision_models.ipynb", "decision_core"),
    ]:
        book = nbformat.read(COHORT / source, 4)
        nbformat.validate(book)
        implementation = "\n\n".join(
            c.source
            for c in book.cells
            if c.cell_type == "code" and "runtime" in c.metadata.get("tags", [])
        )
        ast.parse(implementation)
        (APP / f"backend/{module}.py").write_text(
            '"""Application memory and agent implementation."""\n'
            + implementation
            + "\n"
        )
    print("Synced three application runtimes; interactive UI is maintained separately.")


if __name__ == "__main__":
    build()
