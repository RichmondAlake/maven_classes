"""Build the Memorizz edition (Parts 0–12) from tools/memorizz_lesson.txt.

MemAgent runs the agent loop and Memorizz supplies every memory type on Oracle.
Part 9.1 (the five travel tools) is taken from Part 8.1 of tools/travel_lesson.txt so both
editions share one tool implementation.
"""
import nbformat
from build_notebooks import ROOT, attach_diagrams, parse_lesson
from notebook_setup import add_dependency_setup

MEMORIZZ_PACKAGE = "memorizz==0.17.23"
INCLUDE = "# %% include travel-tools\n"

lesson = (ROOT / "tools/memorizz_lesson.txt").read_text()
travel = (ROOT / "tools/travel_lesson.txt").read_text()
# The from-scratch lesson teaches toolbox memory as Part 8; in this edition it is Part 9.
start = travel.index("# %% [markdown]\n### Part 8.1")
end = travel.index("# %% [markdown]\n### Part 8.2")
tools_section = travel[start:end].replace("### Part 8.1 ·", "### Part 9.1 ·", 1)
if lesson.count(INCLUDE) != 1:
    raise SystemExit("memorizz_lesson.txt needs exactly one travel-tools include.")

cells = parse_lesson(lesson.replace(INCLUDE, tools_section), bridge_consecutive_code=False)
attach_diagrams(cells, ROOT / "data")
cells = add_dependency_setup(cells, "requirements-memorizz.txt")

notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3.12"}
notebook.metadata.memorizz_package = MEMORIZZ_PACKAGE
nbformat.validate(notebook)
nbformat.write(notebook, ROOT / "implementing_memory_aware_agents_memorizz.ipynb")
print(f"Memorizz edition: {len(cells)} cells; {sum(c.cell_type == 'code' for c in cells)} code cells")
