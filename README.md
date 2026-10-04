# Agent Evaluation & Benchmarking Framework

A reproducible harness for comparing ReAct, Plan-and-Execute, LangGraph, a no-tools direct baseline, and optional CrewAI/AutoGen adapters using a shared tool interface.

## What's included

- 130 custom tasks and 40 independently authored GAIA-style tasks. All have deterministic short-answer labels, required task budgets, a dev/test split, and structured grading; 25 custom tasks exercise injected tool faults.
- Shared calculator, restricted Python, fixed-snapshot search, local-corpus retrieval and corpus-only file reader.
- OpenAI-compatible Chat Completions client configured only by environment variables, with SQLite response caching.
- Repeated evaluation with token/step/time budgets, projected spend guard, SQLite/JSONL/CSV persistence and confidence intervals.
- Paired reflection ON/OFF evaluation, with answer-free critique prompts and at most two cycles.
- LangGraph StateGraph with conditional routing and in-memory checkpointing; optional CrewAI and AutoGen single-worker adapters.

## Setup

Python 3.11 or later is supported. Python 3.12 is recommended for optional framework comparisons.

```powershell
uv sync --python 3.12 --extra dev
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env` (or the process environment). For a compatible endpoint, also set `OPENAI_BASE_URL`. Never commit a populated `.env`. The live runner requires a key; benchmark generation, validation, tests and spend estimates do not.

The pinned core versions are in `pyproject.toml`, `requirements.txt` and `uv.lock`: `langgraph==1.2.10`, `openai==2.53.0`, `numpy==2.2.6`, `PyYAML==6.0.3`, `python-dotenv==1.2.2`, and `pytest==9.1.1` for development. Optional framework versions are pinned in `requirements-frameworks.txt` (`crewai==1.15.10`, `autogen-agentchat==0.7.5`, `autogen-ext[openai]==0.7.5`).

## Quick start

```powershell
# Verify the frozen benchmark and estimate the pilot without making API calls
uv run python run_eval.py --agent react --benchmark gaia_style --config configs/experiments/pilot.yaml --estimate-only

# Pilot: 15 dev tasks, four architectures, five repeats
uv run python run_eval.py --agent all --benchmark custom --config configs/experiments/pilot.yaml

# Enable paired reflection ON/OFF runs (still subject to the config spend cap)
uv run python run_eval.py --agent react --benchmark custom --config configs/experiments/pilot.yaml --reflection

# Full run: after the pilot, pass its summary for an observed-usage projection
uv run python run_eval.py --agent all --benchmark all --config configs/experiments/full.yaml --pilot-results results/<pilot-run-id>/summary.json

# Compare framework adapters; install the optional dependencies first
uv sync --python 3.12 --extra dev --extra frameworks
uv run python run_eval.py --framework-compare --benchmark custom --config configs/experiments/pilot.yaml
```

Choose `--model-role development` or `--model-role primary` to override the config's model role. `--split`, `--task-limit`, and `--repeats` override their corresponding experiment values. A non-estimate full run requires `--pilot-results`; the maximum per-agent observed pilot input/output means are projected with a 25% safety uplift before the cap check. `--resume RUN_ID` resumes completed `(agent, task, repeat, setting)` rows in `results/RUN_ID/`, reusing a row only when its config and task fingerprints still match.

## Reproducibility and outputs

`configs/experiments/pilot.yaml` and `full.yaml` capture the model, controls, budgets, repeats, caching, and hard spend cap. API and tool outputs are cached in SQLite when enabled. Search uses `benchmarks/corpus/search_snapshots.json`; retrieval uses `benchmarks/corpus/documents.jsonl`.

Each run writes:

- `results/<run_id>/results.sqlite3` — resumable per-repeat rows.
- `results/<run_id>/results.jsonl` — structured trace and tool I/O log.
- `results/<run_id>/results.csv` and `summary.json` — analysis-friendly results and aggregate metrics.

Run `uv run python -m benchmarks.validate` to check the checked-in suites. Regenerate and freeze both datasets with `uv run python run_eval.py --generate-benchmarks`; `benchmarks/VERSION.json` records their version, counts, validation summary and SHA-256 digests. To summarize a run and create SVG charts, use `uv run python analysis/aggregate_results.py --run <run_id>`.
The analyzer writes success-rate, success-versus-cost, and success-versus-latency charts to the output directory.

Run local validation with `uv run pytest`. CI runs the unit suite, benchmark validator, and a no-credential pilot estimate.

## Experimental controls

- All agent architectures use the configured model, temperature and maximum completion tokens. The pilot uses the development model; the full config defaults to the primary model. The judge model is separate from the evaluated model.
- ReAct, Plan-and-Execute, LangGraph and the framework adapters call the same `ToolBox`. Direct baseline has no tools.
- Tasks define `max_steps`; experiment budgets cap steps, tokens and wall time. Runs exceeding a budget are marked `budget_exhausted`.
- Deterministic exact/normalized and numeric-tolerance graders run locally. Fuzzy grading is deterministic. LLM rubric judging is a separate-model, temperature-zero majority vote and is refused for calibration below 30 hand-labeled items; no LLM-judge benchmark items are included in the frozen suites.
- The harness estimates spend before requests and refuses a run above `spend_cap_usd`. Actual token usage and cost are recorded when providers return usage metadata.
- Bootstrap 95% intervals and paired bootstrap/McNemar comparisons are calculated from per-task repeats. Cached calls are flagged and reported separately from uncached latency.
- Reflection critiques receive task text and execution traces but not expected answers or rubric labels.

## Framework mapping

See [frameworks/README.md](frameworks/README.md) for the one-worker CrewAI and AutoGen mapping, shared tools, instrumentation limitations, and qualitative comparison.

## Safety and limitations

The Python tool runs a syntax-whitelisted subset in a short-lived isolated subprocess with an execution timeout; imports and filesystem/network APIs are not permitted. Windows does not provide a portable standard-library memory limit for this subprocess. Treat this as a benchmark sandbox, not a boundary for hostile multi-tenant code; use a container/OS sandbox with memory and CPU quotas for production.

## Deviations from plan

- The GAIA-style tasks are newly authored and fully local; they are not licensed or copied GAIA data.
- The current environment has no API credentials, so no live pilot, sensitivity run, or empirical framework comparison has been executed. The report intentionally contains no fabricated performance results.
- CrewAI currently fails to import in the installed Python 3.14 environment because its ChromaDB dependency uses incompatible Pydantic v1 internals. AutoGen's OpenAI extension is optional and is not installed in that environment. The optional adapters are pinned and intended for the documented Python 3.12 setup, but their live-provider runs remain unverified here.
- Windows prevents enforcing a portable hard memory limit on the restricted Python subprocess; see the safety note above.
- CrewAI/AutoGen do not provide comparable per-step token usage through the current adapter paths; their token counts are marked as prompt/answer estimates. Do not use those estimates for claims requiring token-level parity until provider usage is wired through.
- A third-party clean-environment live pilot has not been performed. The workflow and CI smoke are offline and credential-free.
