"""Keep kernel dependency installation consistent across the three lessons."""

import nbformat

SETUP_TAG = "dependency-install"


def add_dependency_setup(cells, requirements="requirements.txt"):
    """Insert an installation explanation and cell immediately after the title."""
    cells = [cell for cell in cells if SETUP_TAG not in cell.metadata.get("tags", [])]
    explanation = nbformat.v4.new_markdown_cell(
        "### Install the notebook packages\n\n"
        "Use a **Python 3.11 or 3.12** kernel. Open this notebook from the "
        "`cohort_1` folder so the companion requirements files are available. "
        "Run the next cell before the lesson imports: **%pip** installs into "
        "the Python environment used by this notebook's active kernel.\n\n"
        f"`{requirements}` installs the tested lesson dependencies, including "
        "**sentence-transformers** for local embeddings and reranking, "
        "**oracledb** and **langchain-oracledb** for Oracle memory, "
        "**anthropic** for generation, and **tavily-python** for live search. "
        + (
            "It also installs the published **Memorizz** package. "
            if requirements == "requirements-memorizz.txt"
            else ""
        )
        + "\n\nAfter the first installation finishes, **restart the kernel**, "
        "then run the lesson cells in order. Package installation needs "
        "internet access; it does not need your provider API keys.",
        metadata={"tags": [SETUP_TAG]},
        id="dependency-install-notes",
    )
    installation = nbformat.v4.new_code_cell(
        "# Install packages into this notebook's active Python kernel.\n"
        f"%pip install -r {requirements}",
        metadata={"tags": [SETUP_TAG]},
        id="dependency-install-code",
    )
    cells[1:1] = [explanation, installation]
    return cells
