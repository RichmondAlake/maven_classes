"""Build the reviewed lesson source into notebooks; embed portable PNG attachments."""
import ast
import base64
import hashlib
import re
from pathlib import Path

import nbformat
from notebook_setup import add_dependency_setup

ROOT = Path(__file__).resolve().parents[1]


def read_lesson(path):
    cells = []
    matches = list(re.finditer(r"^# %% (\[markdown\]|runtime|example)\n", path.read_text(), re.M))
    text = path.read_text()

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        source = text[match.end():end].strip()
        if match[1] == "[markdown]":
            cell = nbformat.v4.new_markdown_cell(source)
        else:
            ast.parse(source)
            cell = nbformat.v4.new_code_cell(source, metadata={"tags": [match[1]]})

        if cell.cell_type == "code" and cells and cells[-1].cell_type == "code":
            cells.append(nbformat.v4.new_markdown_cell(
                "### Supply your own trip\n\nThe next cell reads your natural-language trip request "
                "and calls parse_trip. Its structured result is passed explicitly into later RAG "
                "and session-memory calls; no trip is known implicitly. Dates, budget and "
                "constraints come from your input, and missing fields remain unknown."
            ))
        cell.id = hashlib.sha256(f"{index}:{source}".encode()).hexdigest()[:12]
        cells.append(cell)

    return cells


def attach_diagrams(cells, data_dir):
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        names = re.findall(r"attachment:([\w.-]+)", cell.source)
        if names:
            cell.attachments = {
                name: {"image/png": base64.b64encode((data_dir / name).read_bytes()).decode()}
                for name in names
            }


def build_main():
    cells = read_lesson(ROOT / "tools/travel_lesson.txt")
    attach_diagrams(cells, ROOT / "data")
    cells = add_dependency_setup(cells)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    notebook.metadata.lesson_version = 2
    nbformat.validate(notebook)

    path = ROOT / "implementing_memory_aware_agents.ipynb"
    nbformat.write(notebook, path)

    # App startup includes implementation cells, never demonstrations or prompts for a trip.
    implementation = "\n\n".join(cell.source for cell in cells if "runtime" in cell.metadata.get("tags", []))
    ast.parse(implementation)
    (ROOT / "appbook/backend/notebook_core.py").write_text(
        '"""Generated from the reviewed runtime-tagged notebook cells."""\n' + implementation + "\n"
    )
    print(f"Built {len(cells)} cells; {sum(c.cell_type == 'code' for c in cells)} code cells.")


if __name__ == "__main__":
    build_main()
