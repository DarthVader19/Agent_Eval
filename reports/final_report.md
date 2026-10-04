# Agent Evaluation & Benchmarking Framework — Final Report

**Report status:** implementation and offline verification complete; empirical LLM results pending credentials.  
**Benchmark version:** v1.0  
**Python target:** 3.11+ (Python 3.12 recommended for framework adapters)  
**Generated:** 2026-10-04

## Executive summary

The repository now contains a config-driven evaluation harness, four agent modes (ReAct, Plan-and-Execute, LangGraph and direct no-tools), a shared cached/fault-injectable `ToolBox`, repeated SQLite-backed evaluation, deterministic graders, bootstrap confidence intervals, bounded reflection, framework adapters and a versioned local benchmark.

The benchmark validator reports **130 custom cases** and **40 GAIA-style cases**. The custom suite has 25 fault-injection cases and an exact 20% test split. Both benchmark files and validation summaries are hashed in `benchmarks/VERSION.json`.

No API credentials were available in the working environment, so no live model task was attempted. Accordingly, there are no measured success, token, latency, error-recovery, reflection or framework-comparison results to report. This is deliberate: synthetic/random simulator metrics have been removed rather than presented as evidence.

## Offline verification results

| Check | Result |
|---|---:|
| Custom task cases | 130 |
| GAIA-style local cases | 40 |
| Custom injected-fault cases | 25 |
| Custom held-out test split | 20% (26/130) |
| GAIA-style held-out test split | 20% (8/40) |
| API runs completed | 0 |
| API spend | $0.00 |
| Offline unit tests | 23 passed |

The initially failing fuzzy-grader case is fixed by normalizing punctuation and whitespace before scoring. Unit tests cover the graders, LLM-judge calibration token accounting, restricted calculator/Python tool, tool I/O and fault modes, budget exhaustion, real LangGraph state-graph execution, reflection ground-truth exclusion, spend-cap blocking, SQLite repeats/resume and paired statistics.

## Pilot budget estimate

The checked-in pilot config selects 15 dev cases, four agent architectures and five repeats. At the configured per-run ceiling of 3,200 input plus 800 output tokens with `gpt-4o-mini`, the projected agent-model total is **$0.288**. This conservative no-usage estimate is used before the pilot. A full run requires the pilot's saved token-use summary; its estimate uses the highest per-agent pilot input/output means with a 25% safety uplift. Actual cost remains **$0.00** because no model calls were made.

| Pilot configuration | Value |
|---|---:|
| Tasks | 15 |
| Agent architectures | 4 |
| Repeats | 5 |
| Projected runs | 300 |
| Assumed tokens per run | 3,200 input / 800 output |
| Estimated cost | $0.288 |
| Spend cap | $5.00 |

Run `python run_eval.py --agent all --benchmark custom --config configs/experiments/pilot.yaml --estimate-only` to reproduce the estimate without credentials. The CLI refuses to start if the estimate exceeds the configured hard cap or a model has no configured token prices.

## Evaluation methodology

- **Agent modes:** ReAct tool loop; plan-first Plan-and-Execute with replanning context after errors; a compiled LangGraph `StateGraph` with a conditional continue/end transition and in-memory checkpointing; and a single-call direct LLM baseline without tools.
- **Shared controls:** configured model, temperature (0.0 default), completion limit, shared `ToolBox`, per-task step/token/time budgets, and configurable response caching.
- **Tools:** safe arithmetic evaluator; syntax-whitelisted Python subprocess; snapshot-based search; fixed-corpus retriever and corpus-only file reader. Faults are task-scoped and include transient, permanent, timeout, empty, malformed and plausible-wrong output.
- **Grading:** normalized exact and numeric tolerance for frozen short-answer tasks. Fuzzy grading is deterministic. LLM rubric judgment uses a separate configured model, temperature 0, majority vote, and a minimum 30 hand-labeled calibration sample with Cohen's kappa; do not use it at scale unless calibration passes.
- **Repeats and uncertainty:** pilot 5 repeats; full run 3. Bootstrap 95% intervals, task-level all-repeats-correct rate, paired difference intervals and exact paired binary comparisons are persisted.
- **Persistence:** each task-repeat and tool input/output is logged in SQLite and JSONL, with CSV and summary JSON for analysis; model and tool caches are SQLite-backed.
- **Reflection:** max two cycles. The critic receives the task and trace but no expected answer/rubric; reflected and non-reflected results are paired by task/repeat, with delta, extra tokens/latency, regression and rescue rates.

## Empirical results

| Metric | Result | 95% CI | Status |
|---|---:|---:|---|
| Success rate | n/a | n/a | Live pilot not run |
| Tokens per successful task | n/a | n/a | Live pilot not run |
| Error recovery rate | n/a | n/a | Live pilot not run |
| Mean / p50 / p95 latency | n/a | n/a | Live pilot not run |
| Reflection success delta | n/a | n/a | Paired live runs not run |
| Framework success comparison | n/a | n/a | Optional adapters not live-run |

No production architecture recommendation is inferred from absent measurements. Use LangGraph where explicit state/checkpoint/conditional control is a requirement; use ReAct for a simple instrumentable loop and direct baseline to measure tool value. Treat these as design guidance, not measured ranking.

## Framework comparison

The CrewAI and AutoGen adapters implement one-worker mappings over the same shared `ToolBox`; the mapping and qualitative (implementation-time, not measured) assessment are in [frameworks/README.md](../frameworks/README.md). Runtime validation was blocked in the current Python 3.14 environment: CrewAI's installed ChromaDB dependency raises a Pydantic-v1 compatibility error, and `autogen-ext[openai]` is not installed there. The optional dependencies are pinned, and Python 3.12 is the recommended environment, but no framework performance comparison is claimed.

## Limitations and deviations

1. The 40 locally authored tasks are GAIA-style, not official GAIA items; they use fixed local facts and short verifiable answers.
2. No live pilot or clean-environment live rerun was possible without provider credentials.
3. No second-model sensitivity result or hand-labeled LLM-judge calibration result is reported. The frozen suites use deterministic graders, so the judge is not invoked for headline scores.
4. CrewAI and AutoGen currently expose prompt/answer token estimates in the adapters, not usage metadata at each model turn; these estimates are not adequate for final token-parity claims.
5. The Python tool disallows imports and filesystem/network syntax and enforces a timeout, but portable hard memory quotas are not provided on Windows. Use Docker/OS sandboxing for hostile code in production.
6. CI validates offline code, task schemas and pilot cost estimation. It cannot demonstrate provider response quality or the clean-environment live pilot target without a separately provisioned API key.

## Reproduction

See [README.md](../README.md) for Python setup, pinned dependencies, environment-variable configuration, pilot/full commands, resume instructions and test commands. The source of truth for run controls is in `configs/experiments/`; benchmark counts, validation and digests are in `benchmarks/VERSION.json`.
