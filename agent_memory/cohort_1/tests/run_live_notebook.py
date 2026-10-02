"""Execute all cells with real providers and isolated Oracle storage.

Credentials stay in memory. --env-file is an explicitly selected private file;
its values are never copied to notebook source or validation output.
"""
import argparse
import base64
import contextlib
import getpass
import hashlib
import io
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import nbformat
from dotenv import dotenv_values

parser = argparse.ArgumentParser()
parser.add_argument("notebook", type=Path)
parser.add_argument("--env-file", type=Path)
parser.add_argument("--request", required=True)
parser.add_argument("--policy-query", required=True)
parser.add_argument("--live-query", required=True)
parser.add_argument("--preference", required=True)
parser.add_argument("--report", type=Path)
parser.add_argument("--context-limit", default="12000")
args = parser.parse_args()
source = args.notebook.resolve()
report_path = args.report.resolve() if args.report else source.with_name(source.stem + ".validation.json")
env_keys = dotenv_values(args.env_file) if args.env_file else {}
secrets_in_memory = []

for name in ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]:
    value = os.getenv(name) or env_keys.get(name) or getpass.getpass(f"{name} (hidden): ").strip()
    if not value:
        raise ValueError(f"{name} is required.")
    os.environ[name] = value
    secrets_in_memory.append(value)

for name in ["TYPESAFE_API_KEY", "VOYAGE_API_KEY"]:
    value = os.getenv(name) or env_keys.get(name)
    if value:
        os.environ[name] = value
        secrets_in_memory.append(value)

schema = "AMR_" + secrets.token_hex(5).upper()
tablespace = schema + "_TS"
password = "R_" + secrets.token_hex(18)
secrets_in_memory.append(password)
container = "acme-oracle-free"
nb = nbformat.read(source, as_version=4)
nbformat.validate(nb)
namespace = {"__name__": "__main__"}
completed = []
started = time.perf_counter()
created = False
active_outputs = []


def scrub(value):
    text = str(value)
    for secret in secrets_in_memory:
        text = text.replace(secret, "[redacted]")
    return re.sub(r"(sk-ant-|tvly-)[A-Za-z0-9_-]+", "[redacted]", text)


def admin(sql):
    result = subprocess.run(
        ["/usr/local/bin/docker", "exec", "-i", container, "sqlplus", "-s", "/", "as", "sysdba"],
        input="WHENEVER SQLERROR EXIT SQL.SQLCODE\nALTER SESSION SET CONTAINER=FREEPDB1;\n" + sql + "\nexit\n",
        text=True, capture_output=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError(scrub(result.stdout + result.stderr))


def capture_display(*values, **kwargs):
    for value in values:
        if hasattr(value, "_repr_html_"):
            data = {"text/html": scrub(value._repr_html_()), "text/plain": scrub(str(value))}
        else:
            data = {"text/plain": scrub(str(value))}
        active_outputs.append(nbformat.v4.new_output("display_data", data=data, metadata={}))


# Capture plot outputs in the executed artifact, just as a notebook kernel does.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def capture_plots(*args, **kwargs):
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format="png", bbox_inches="tight")
            active_outputs.append(nbformat.v4.new_output("display_data", data={
                "image/png": base64.b64encode(buffer.getvalue()).decode(),
            }, metadata={}))
        plt.close("all")

    plt.show = capture_plots
except ImportError:
    pass


os.environ.update(
    ORACLE_USER=schema, ORACLE_PASSWORD=password,
    ORACLE_DSN="127.0.0.1:1521/FREEPDB1",
    NOTEBOOK_USE_ENV_KEYS="1", ANTHROPIC_MODEL="claude-opus-5-5",
    TRAVEL_REQUEST=args.request, TRAVEL_POLICY_QUERY=args.policy_query,
    TRAVEL_LIVE_QUERY=args.live_query, TRAVEL_PREFERENCE_TEXT=args.preference,
    TRAVEL_CONTEXT_LIMIT=args.context_limit,
    HF_HOME="/private/tmp/maven_hf_cache", HF_HUB_DISABLE_XET="1",
    TOKENIZERS_PARALLELISM="false", ANTHROPIC_BASE_URL="https://api.anthropic.com",
)
os.chdir(source.parent)
report = {
    "status": "running", "source": source.name,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "model": "claude-opus-5-5", "credentials_saved": False,
    "oracle": "isolated Oracle 26ai schema with HNSW vector pool",
    "validation_request": args.request,
}

try:
    admin(
        f"CREATE TABLESPACE {tablespace} DATAFILE '/opt/oracle/oradata/{tablespace}.dbf' SIZE 16M AUTOEXTEND ON NEXT 16M MAXSIZE 256M;\n"
        f'CREATE USER {schema} IDENTIFIED BY "{password}" DEFAULT TABLESPACE {tablespace} QUOTA 256M ON {tablespace};\n'
        f"GRANT CREATE SESSION, CREATE TABLE, CREATE SEQUENCE TO {schema};"
    )
    created = True
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        output = io.StringIO()
        active_outputs = []
        cell.execution_count = len(completed) + 1
        print(f"Running cell {index}: {cell.source.splitlines()[0][:85]}", flush=True)
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                if "dependency-install" in cell.metadata.get("tags", []):
                    # Match %pip: install with this runner's Python interpreter.
                    command = next(line for line in cell.source.splitlines() if line.startswith("%pip "))
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", *shlex.split(command[5:])],
                        capture_output=True, text=True,
                    )
                    print(result.stdout + result.stderr)
                    result.check_returncode()
                else:
                    exec(compile(cell.source, f"{source.name}:cell-{index}", "exec"), namespace)
                namespace["display"] = capture_display

            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=scrub(output.getvalue()))]
            cell.outputs += active_outputs
            completed.append(index)
            if len(output.getvalue()) and any(term in cell.source for term in ["agent_result", "plan_text", "cache reads", "compaction"]):
                print(scrub(output.getvalue())[:1600], flush=True)

        except Exception as error:
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=scrub(output.getvalue()))] + active_outputs
            cell.outputs.append(nbformat.v4.new_output(
                "error", ename=type(error).__name__, evalue=scrub(error),
                traceback=[scrub(line) for line in traceback.format_exception(error)],
            ))
            raise

    report.update(
        status="passed", code_cells_executed=len(completed), total_cells=len(nb.cells),
        calls=namespace.get("CALLS", []), searches=namespace.get("SEARCH_CALLS", []),
        hnsw_plan=namespace.get("plan_text"),
        provider_cache_read_tokens=sum(record.get("cache_read_tokens", 0) for record in namespace.get("CALLS", [])),
        estimated_llm_usd=sum(record.get("estimated_usd") or 0 for record in namespace.get("CALLS", [])),
        embedding_calls=namespace.get("EMBEDDING_CALLS", []),
        measurements=namespace.get("CACHE_METRICS", []),
        stages=namespace.get("stage_results", []),
        decision_calls=namespace.get("DECISION_CALLS", []),
    )

except Exception as error:
    report.update(status="failed", error_type=type(error).__name__, error=scrub(error), completed_cells=completed)
    report['calls'] = namespace.get('CALLS', [])
    report['decision_calls'] = namespace.get('DECISION_CALLS', [])
    report['agent_inspection'] = namespace.get('LAST_AGENT_RUN')
    if namespace.get('workflow_memory'):
        with contextlib.suppress(Exception):
            report['workflow_tail'] = namespace['workflow_memory'](20)
    print(scrub(traceback.format_exc()), flush=True)

finally:
    provider = namespace.get("provider")
    if provider:
        with contextlib.suppress(Exception):
            provider.close()
    for name in ["conn", "client"]:
        value = namespace.get(name)
        if value:
            with contextlib.suppress(Exception):
                value.close()
    if created:
        admin(f"DROP USER {schema} CASCADE;\nDROP TABLESPACE {tablespace} INCLUDING CONTENTS AND DATAFILES;")
    report["seconds"] = round(time.perf_counter() - started, 2)
    executed = source.with_name(source.stem + ".executed.ipynb")
    serialized = scrub(nbformat.writes(nb))
    executed.write_text(serialized)
    report_path.write_text(scrub(json.dumps(report, indent=2, default=str)) + "\n")
    print("RESULT:", report["status"], "| completed code cells:", len(completed), flush=True)

raise SystemExit(0 if report["status"] == "passed" else 1)
