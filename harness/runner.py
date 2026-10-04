from __future__ import annotations

import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from agents.base import Budget
from agents.direct_baseline import DirectBaselineAgent
from agents.langgraph_agent import LangGraphAgent
from agents.plan_execute_agent import PlanExecuteAgent
from agents.react_agent import ReActAgent
from harness.judge import LLMJudge
from harness.metrics import MetricsCalculator
from harness.model import CostEstimator
from tools.toolbox import ToolBox
from frameworks.autogen_runner import AutoGenAdapter
from frameworks.crewai_runner import CrewAIAdapter


AGENT_TYPES = {
    "react": ReActAgent,
    "plan_execute": PlanExecuteAgent,
    "langgraph": LangGraphAgent,
    "direct": DirectBaselineAgent,
    "crewai": CrewAIAdapter,
    "autogen": AutoGenAdapter,
}


class EvaluationHarness:
    """Repeated, resumable evaluation with budgets and SQLite/JSONL trace persistence."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.metrics_calculator = MetricsCalculator()
        judge_config = dict(config.get("judge", config.get("judge_config", {})))
        if judge_config:
            judge_config.setdefault("cache_enabled", bool(config.get("cache", {}).get("enabled", True)))
        self.judge = LLMJudge(judge_config) if judge_config else LLMJudge({"model": config.get("model", "gpt-4o")})
        self.judge_calibration_report: dict[str, Any] | None = None
        self.judge_calibration_cost_usd = 0.0
        self.run_id = str(config.get("resume_run") or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}")
        self.results_dir = Path(config.get("results_dir", Path("results") / self.run_id))
        self.database_path = self.results_dir / "results.sqlite3"

    def run_evaluation(
        self,
        agents_config: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
        parallel: bool = False,
        max_workers: int = 4,
        repeats: int | None = None,
        *,
        estimate_only: bool = False,
    ) -> dict[str, Any]:
        repeat_count = int(repeats if repeats is not None else self.config.get("repeats", 1))
        if repeat_count < 1:
            raise ValueError("repeats must be a positive integer.")
        if not agents_config or not tasks:
            raise ValueError("At least one agent and one task are required.")
        estimate = self.estimate_spend(agents_config, tasks, repeat_count)
        cap = float(self.config.get("spend_cap_usd", 0.0))
        if cap > 0 and estimate["projected_cost_usd"] > cap:
            raise RuntimeError(
                f"Projected cost ${estimate['projected_cost_usd']:.4f} exceeds configured spend cap "
                f"${cap:.2f}; no model requests were made."
            )
        if estimate_only:
            return {"estimate": estimate, "results": [], "summary": {}}

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self._calibrate_judge_if_needed(agents_config, tasks)
        combinations = [
            (agent_config, task, repeat)
            for agent_config in agents_config
            for task in tasks
            for repeat in range(1, repeat_count + 1)
        ]
        results: list[dict[str, Any]] = []
        pending = []
        reflection_enabled = bool(self.config.get("reflection", {}).get("enabled", False))
        for agent_config, task, repeat in combinations:
            saved = self._load_saved(agent_config["type"], task["id"], repeat)
            saved_on = self._load_saved(agent_config["type"], task["id"], repeat, "on") if reflection_enabled else None
            if saved is not None and not _matches_current(saved, agent_config, task, self.config):
                saved = None
            if saved_on is not None and not _matches_current(saved_on, agent_config, task, self.config):
                saved_on = None
            if saved is not None and (not reflection_enabled or saved_on is not None):
                results.append(saved)
            else:
                pending.append((agent_config, task, repeat))
        if parallel and pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_one, agent_config, task, repeat): (agent_config, task, repeat)
                    for agent_config, task, repeat in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    self._persist_one(result)
                    results.append(result)
        else:
            for agent_config, task, repeat in pending:
                result = self._run_one(agent_config, task, repeat)
                self._persist_one(result)
                results.append(result)
        if reflection_enabled:
            results = self._run_reflection_pairs(results, agents_config, tasks)
        results.sort(key=lambda row: (row["agent_type"], row["task_id"], row["repeat"]))
        summary = self.metrics_calculator.calculate_summary(results)
        summary["run"] = {
            "run_id": self.run_id,
            "benchmark_cases": len(tasks),
            "repeats": repeat_count,
            "estimated_cost_usd": estimate["projected_cost_usd"],
            "actual_cost_usd": _actual_spend(results) + self.judge_calibration_cost_usd,
            "judge_calibration_cost_usd": self.judge_calibration_cost_usd,
            "judge_calibration": self.judge_calibration_report,
            "cost_estimate": estimate,
        }
        if reflection_enabled:
            summary["reflection"] = self._reflection_summary(results)
        if not reflection_enabled:
            results = [
                {key: value for key, value in row.items() if not key.startswith("_")}
                for row in results
            ]
        self._save_results(results, summary)
        return {"detailed_results": results, "summary": summary, "estimate": estimate}

    def estimate_spend(
        self, agents_config: list[dict[str, Any]], tasks: list[dict[str, Any]], repeats: int
    ) -> dict[str, Any]:
        default_model = str(self.config.get("model", "gpt-4o-mini"))
        budget = self.config.get("budgets", {})
        max_tokens = int(budget.get("max_tokens", self.config.get("max_tokens", 4000)))
        pilot_usage = self.config.get("pilot_usage", {})
        if pilot_usage:
            uplift = float(pilot_usage.get("safety_uplift", 1.25))
            input_tokens = int(float(pilot_usage["input_tokens"]) * uplift)
            output_tokens = int(float(pilot_usage["output_tokens"]) * uplift)
            estimate_basis = "pilot_mean_with_safety_uplift"
        else:
            input_tokens = int(max_tokens * 0.8)
            output_tokens = max_tokens - input_tokens
            estimate_basis = "configured_budget_ceiling"
        prices = {
            name: tuple(values)
            for name, values in self.config.get("pricing_per_million_tokens", {}).items()
        }
        model_costs: dict[str, float] = {}
        for agent_config in agents_config:
            model = str(agent_config.get("model", default_model))
            if self.config.get("spend_cap_usd", 0) and model not in prices:
                raise ValueError(f"No token pricing configured for model {model!r}; refusing to evaluate with a spend cap.")
            model_costs[model] = model_costs.get(model, 0.0) + CostEstimator(prices).estimate(
                model, len(tasks), 1, repeats, input_tokens, output_tokens
            )
        judge_tasks = sum(
            task.get("grader_type") in {"llm_judge", "rubric"}
            or task.get("rubric", {}).get("type") in {"llm_judge", "rubric"}
            for task in tasks
        )
        judge_cost = 0.0
        judge_model = str(self.config.get("judge", {}).get("model", ""))
        if judge_tasks and judge_model:
            if self.config.get("spend_cap_usd", 0) and judge_model not in prices:
                raise ValueError(f"No token pricing configured for judge model {judge_model!r}; refusing to evaluate.")
            votes = max(1, int(self.config.get("judge", {}).get("votes", 3)))
            judge_cost = CostEstimator(prices).estimate(
                judge_model,
                judge_tasks,
                len(agents_config) * repeats * votes,
                1,
                512,
                int(self.config.get("judge", {}).get("max_output_tokens", 256)),
            )
            judge_config = self.config.get("judge", {})
            calibration_samples = int(judge_config.get("calibration_sample_count", 0))
            if not calibration_samples:
                calibration_samples = len(self.config.get("judge_calibration_samples", []))
            calibration_file = judge_config.get("calibration_file")
            if not calibration_samples and calibration_file and Path(calibration_file).exists():
                calibration_samples = sum(
                    bool(line.strip()) for line in Path(calibration_file).read_text(encoding="utf-8").splitlines()
                )
            if calibration_samples:
                judge_cost += CostEstimator(prices).estimate(
                    judge_model,
                    calibration_samples,
                    max(1, int(self.config.get("judge", {}).get("votes", 3))),
                    1,
                    512,
                    int(self.config.get("judge", {}).get("max_output_tokens", 256)),
                )
        estimate = sum(model_costs.values()) + judge_cost
        return {
            "models": model_costs,
            "judge_model": judge_model or None,
            "judge_estimated_cost_usd": judge_cost,
            "tasks": len(tasks),
            "agents": len(agents_config),
            "repeats": repeats,
            "assumed_input_tokens_per_run": input_tokens,
            "assumed_output_tokens_per_run": output_tokens,
            "basis": estimate_basis,
            "pilot_usage": pilot_usage or None,
            "projected_cost_usd": estimate,
        }

    def _calibrate_judge_if_needed(
        self, agents_config: list[dict[str, Any]], tasks: list[dict[str, Any]]
    ) -> None:
        needs_judge = any(
            task.get("grader_type") in {"llm_judge", "rubric"}
            or task.get("rubric", {}).get("type") in {"llm_judge", "rubric"}
            for task in tasks
        )
        if not needs_judge:
            return
        judge_config = self.config.get("judge", {})
        judge_model = str(judge_config.get("model", ""))
        agent_models = {
            str(agent.get("model", self.config.get("model", ""))) for agent in agents_config
        }
        if not judge_model or judge_model in agent_models:
            raise ValueError("LLM judge must use a configured model different from every evaluated agent model.")
        samples = self.config.get("judge_calibration_samples")
        calibration_file = judge_config.get("calibration_file")
        if samples is None and calibration_file:
            path = Path(calibration_file)
            samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not samples or len(samples) < 30:
            raise RuntimeError("LLM-judge scoring requires at least 30 hand-labeled calibration samples.")
        judge_config["calibration_sample_count"] = len(samples)
        report = self.judge.calibrate(samples)
        (self.results_dir / "judge_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report.get("passed"):
            raise RuntimeError(f"LLM judge failed calibration; refusing to score at scale: {report}")
        self.judge_calibration_report = report
        prices = self.config.get("pricing_per_million_tokens", {}).get(judge_model, [0.0, 0.0])
        self.judge_calibration_cost_usd = (
            int(report.get("billed_tokens_in", 0)) * float(prices[0])
            + int(report.get("billed_tokens_out", 0)) * float(prices[1])
        ) / 1_000_000

    def _run_one(self, agent_config: dict[str, Any], task: dict[str, Any], repeat: int) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        fault_config = task.get("fault_config") or {}
        cache_enabled = bool(self.config.get("cache", {}).get("enabled", True)) and not bool(fault_config)
        agent_options = {
            **self.config,
            **agent_config,
            "cache_enabled": cache_enabled,
            "cache_path": str(self.results_dir / "cache.sqlite3"),
            "request_timeout": min(
                float(self.config.get("budgets", {}).get("max_seconds", 90)),
                float(self.config.get("request_timeout", 60)),
            ),
        }
        toolbox = ToolBox(
            cache_enabled=cache_enabled,
            cache_path=self.results_dir / "tool_cache.sqlite3",
            fault_config=fault_config,
            allowed_tools=self.config.get("tools"),
            corpus_path=self.config.get("corpus_path", "benchmarks/corpus/documents.jsonl"),
            search_snapshot=self.config.get("search_snapshot", "benchmarks/corpus/search_snapshots.json"),
        )
        budget = dict(self.config.get("budgets", {}))
        if task.get("max_steps"):
            budget["max_steps"] = min(int(task["max_steps"]), int(budget.get("max_steps", task["max_steps"])))
        agent = self._create_agent(agent_options)
        retries = max(0, int(self.config.get("retries", 0)))
        trace = None
        error = None
        retry_count = 0
        retry_steps = []
        retry_elapsed = 0.0
        retry_tokens = 0
        for attempt in range(retries + 1):
            remaining_steps = int(budget.get("max_steps", 10)) - sum(
                step.step_type in {"reason", "plan", "tool", "model", "final"} for step in retry_steps
            )
            remaining_tokens = int(budget.get("max_tokens", 4000)) - retry_tokens
            remaining_seconds = float(budget.get("max_seconds", 90)) - retry_elapsed
            if min(remaining_steps, remaining_tokens) <= 0 or remaining_seconds <= 0:
                if trace is not None:
                    trace.failure_reason = "budget_exhausted"
                break
            try:
                trace = agent.run(
                    task,
                    toolbox,
                    Budget(
                        max_steps=remaining_steps,
                        max_tokens=remaining_tokens,
                        max_seconds=remaining_seconds,
                    ),
                )
                error = None
                if trace.failure_reason != "model_error" or attempt == retries:
                    retry_count = attempt
                    break
                retry_count = attempt + 1
                retry_steps.extend(trace.steps)
                retry_tokens += trace.total_tokens
                retry_elapsed += trace.total_time
            except Exception as exc:
                error = str(exc)
                if attempt == retries:
                    raise RuntimeError(
                        f"Evaluation failed for agent={agent_config['type']} task={task['id']} repeat={repeat}: {exc}"
                    ) from exc
        if trace is None:
            raise RuntimeError(f"Task budget exhausted before an attempt could start: {task['id']}")
        if retry_steps:
            trace.steps = [*retry_steps, *trace.steps]
            trace.total_tokens += retry_tokens
            trace.total_time += retry_elapsed
            trace.errors = [step.error for step in trace.steps if step.error]
        score = self.judge.evaluate(task, str(trace.final_answer))
        passed = bool(score["passed"])
        metrics = self.metrics_calculator.calculate_task_metrics(trace, score, task)
        judge_usage = score.get("judge_usage", {})
        _include_judge_usage(metrics, judge_usage)
        prices = self.config.get("pricing_per_million_tokens", {}).get(agent_options.get("model", "gpt-4o-mini"), [0.0, 0.0])
        billed_input = sum(step.input_tokens for step in trace.steps if step.step_type in {"reason", "model", "final", "reflection"} and not step.metadata.get("cached"))
        billed_output = sum(step.output_tokens for step in trace.steps if step.step_type in {"reason", "model", "final", "reflection"} and not step.metadata.get("cached"))
        cost = (billed_input * float(prices[0]) + billed_output * float(prices[1])) / 1_000_000
        judge_prices = self.config.get("pricing_per_million_tokens", {}).get(
            str(self.config.get("judge", {}).get("model", "")), [0.0, 0.0]
        )
        cost += (
            int(judge_usage.get("billed_tokens_in", 0)) * float(judge_prices[0])
            + int(judge_usage.get("billed_tokens_out", 0)) * float(judge_prices[1])
        ) / 1_000_000
        fault = next(iter(fault_config.values()), {})
        fault_type = fault if isinstance(fault, str) else fault.get("type", "none")
        return {
            "agent_type": str(agent_config["type"]),
            "agent_config": {key: value for key, value in agent_config.items() if key != "llm_client"},
            "config_hash": _config_hash(agent_config, self.config),
            "task_hash": _task_hash(task),
            "task_id": str(task["id"]),
            "task_category": task.get("category", "unknown"),
            "task_difficulty": task.get("difficulty", "unknown"),
            "fault_type": fault_type,
            "repeat": repeat,
            "setting": "off",
            "success": passed,
            "score": score,
            "metrics": metrics,
            "trace": trace.to_dict(),
            "tool_calls": toolbox.call_log,
            "failure_reason": trace.failure_reason,
            "cost_usd": cost,
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "retry_count": retry_count,
            "_trace_object": trace,
            "_agent_object": agent,
            "_toolbox_object": toolbox,
            "_budget_object": Budget(**budget),
        }

    def _create_agent(self, config: dict[str, Any]):
        agent_type = str(config["type"]).lower()
        try:
            agent_class = AGENT_TYPES[agent_type]
        except KeyError as exc:
            raise ValueError(f"Unknown agent type {agent_type!r}; expected one of {sorted(AGENT_TYPES)}.") from exc
        return agent_class(config)

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS task_runs ("
                "agent_type TEXT NOT NULL, task_id TEXT NOT NULL, repeat INTEGER NOT NULL, "
                "setting TEXT NOT NULL DEFAULT 'off', result_json TEXT NOT NULL, "
                "PRIMARY KEY(agent_type, task_id, repeat, setting))"
            )

    def _load_saved(self, agent_type: str, task_id: str, repeat: int, setting: str = "off") -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT result_json FROM task_runs WHERE agent_type=? AND task_id=? AND repeat=? AND setting=?",
                (agent_type, task_id, repeat, setting),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _persist_one(self, result: dict[str, Any]) -> None:
        payload = json.dumps(
            {key: value for key, value in result.items() if not key.startswith("_")},
            ensure_ascii=False,
            default=str,
        )
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO task_runs(agent_type, task_id, repeat, setting, result_json) VALUES (?, ?, ?, ?, ?)",
                (result["agent_type"], result["task_id"], result["repeat"], result["setting"], payload),
            )
        with (self.results_dir / "results.jsonl").open("a", encoding="utf-8") as file:
            file.write(payload + "\n")

    def _save_results(self, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
        (self.results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        flat = [
            {
                "agent_type": row["agent_type"],
                "task_id": row["task_id"],
                "repeat": row["repeat"],
                "setting": row.get("setting", "off"),
                "category": row["task_category"],
                "difficulty": row["task_difficulty"],
                "fault_type": row["fault_type"],
                "success": row["success"],
                "score": row["score"]["score"],
                "tokens_in": row["metrics"]["tokens_in"],
                "tokens_out": row["metrics"]["tokens_out"],
                "latency_seconds": row["metrics"]["total_latency"],
                "failure_reason": row["failure_reason"],
                "cost_usd": row["cost_usd"],
            }
            for row in results
        ]
        if flat:
            with (self.results_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)

    def _run_reflection_pairs(
        self,
        results: list[dict[str, Any]],
        agents_config: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        from harness.reflection import ReflectionLoop

        loop = ReflectionLoop(self.config.get("reflection", {}))
        indexed_tasks = {str(task["id"]): task for task in tasks}
        paired: list[dict[str, Any]] = []
        for off in results:
            clean_off = {key: value for key, value in off.items() if not key.startswith("_")}
            clean_off["setting"] = "off"
            paired.append(clean_off)
            saved_on = self._load_saved(off["agent_type"], off["task_id"], off["repeat"], "on")
            agent_config = next(item for item in agents_config if item["type"] == off["agent_type"])
            if saved_on is not None and _matches_current(saved_on, agent_config, indexed_tasks[off["task_id"]], self.config):
                paired.append(saved_on)
                continue
            task = indexed_tasks[off["task_id"]]
            trace = off.get("_trace_object")
            agent = off.get("_agent_object")
            toolbox = off.get("_toolbox_object")
            budget = off.get("_budget_object")
            if trace is None or agent is None or toolbox is None or budget is None:
                continue
            reflection_usage: list[dict[str, int]] = []

            def evaluate_reflection_answer(answer: str) -> dict[str, Any]:
                evaluation = self.judge.evaluate(task, answer)
                usage = evaluation.get("judge_usage")
                if usage:
                    reflection_usage.append(usage)
                return evaluation

            reflection = loop.reflect(
                agent,
                task,
                trace,
                toolbox=toolbox,
                budget=budget,
                evaluator=evaluate_reflection_answer,
            )
            score = reflection.score
            reflected_trace = reflection.trace
            metrics = self.metrics_calculator.calculate_task_metrics(reflected_trace, score, task)
            judge_usage: dict[str, int] = {
                "tokens_in": 0, "tokens_out": 0, "billed_tokens_in": 0, "billed_tokens_out": 0
            }
            for usage in reflection_usage:
                _accumulate_usage(judge_usage, usage)
            _include_judge_usage(metrics, judge_usage)
            prices = self.config.get("pricing_per_million_tokens", {}).get(agent_config.get("model", self.config.get("model", "")), [0.0, 0.0])
            judge_prices = self.config.get("pricing_per_million_tokens", {}).get(
                str(self.config.get("judge", {}).get("model", "")), [0.0, 0.0]
            )
            billed_model_tokens_in = sum(step.input_tokens for step in reflected_trace.steps if step.step_type in {"reason", "model", "final", "reflection"} and not step.metadata.get("cached"))
            billed_model_tokens_out = sum(step.output_tokens for step in reflected_trace.steps if step.step_type in {"reason", "model", "final", "reflection"} and not step.metadata.get("cached"))
            reflection_cost = (
                billed_model_tokens_in * float(prices[0])
                + billed_model_tokens_out * float(prices[1])
                + int(judge_usage["billed_tokens_in"]) * float(judge_prices[0])
                + int(judge_usage["billed_tokens_out"]) * float(judge_prices[1])
            ) / 1_000_000
            row = {
                **clean_off,
                "setting": "on",
                "success": bool(score["passed"]),
                "score": score,
                "metrics": metrics,
                "trace": reflected_trace.to_dict(),
                "tool_calls": toolbox.call_log,
                "failure_reason": reflected_trace.failure_reason,
                "cost_usd": reflection_cost,
                "reflection": {
                    "original_score": reflection.original_score,
                    "reflected_score": reflection.reflected_score,
                    "extra_tokens": reflection.tokens_used,
                    "extra_latency_seconds": reflection.time_spent,
                },
            }
            self._persist_one(row)
            paired.append(row)
        return paired

    def _reflection_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        from harness.stats import paired_binary_test, paired_bootstrap_ci

        groups: dict[str, dict[tuple[str, int, str], dict[str, Any]]] = {}
        for row in results:
            key = (row["task_id"], row["repeat"], row["setting"])
            groups.setdefault(row["agent_type"], {})[key] = row
        summary: dict[str, Any] = {}
        for agent_type, rows in groups.items():
            off = {(task_id, repeat): row for (task_id, repeat, setting), row in rows.items() if setting == "off"}
            on = {(task_id, repeat): row for (task_id, repeat, setting), row in rows.items() if setting == "on"}
            common = sorted(off.keys() & on.keys())
            if not common:
                continue
            off_success = [bool(off[key]["success"]) for key in common]
            on_success = [bool(on[key]["success"]) for key in common]
            delta, low, high = paired_bootstrap_ci(off_success, on_success)
            summary[agent_type] = {
                "paired_runs": len(common),
                "success_delta": delta,
                "success_delta_ci95": [low, high],
                "regression_rate": sum(a and not b for a, b in zip(off_success, on_success)) / len(common),
                "rescue_rate": sum(not a and b for a, b in zip(off_success, on_success)) / len(common),
                "paired_binary_test": paired_binary_test(off_success, on_success),
                "extra_tokens_mean": sum(on[key]["metrics"]["total_tokens"] - off[key]["metrics"]["total_tokens"] for key in common) / len(common),
                "extra_latency_mean_seconds": sum(on[key]["metrics"]["total_latency"] - off[key]["metrics"]["total_latency"] for key in common) / len(common),
            }
        return summary


def _actual_spend(results: list[dict[str, Any]]) -> float:
    by_pair = {
        (row["agent_type"], row["task_id"], row["repeat"]): row
        for row in results if row.get("setting") == "off"
    }
    total = 0.0
    for row in results:
        cost = float(row.get("cost_usd", 0.0))
        if row.get("setting") == "on":
            off = by_pair.get((row["agent_type"], row["task_id"], row["repeat"]))
            if off:
                cost = max(0.0, cost - float(off.get("cost_usd", 0.0)))
        total += cost
    return total


def _include_judge_usage(metrics: dict[str, Any], usage: dict[str, Any]) -> None:
    tokens_in = int(usage.get("tokens_in", 0))
    tokens_out = int(usage.get("tokens_out", 0))
    metrics["judge_tokens_in"] += tokens_in
    metrics["judge_tokens_out"] += tokens_out
    metrics["tokens_in"] += tokens_in
    metrics["tokens_out"] += tokens_out
    metrics["total_tokens"] += tokens_in + tokens_out
    metrics["billed_tokens_in"] += int(usage.get("billed_tokens_in", 0))
    metrics["billed_tokens_out"] += int(usage.get("billed_tokens_out", 0))


def _accumulate_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in ("tokens_in", "tokens_out", "billed_tokens_in", "billed_tokens_out"):
        total[key] += int(usage.get(key, 0))


def _config_hash(agent_config: dict[str, Any], config: dict[str, Any]) -> str:
    relevant = {
        "agent": {key: value for key, value in agent_config.items() if key != "llm_client"},
        "model": config.get("model"),
        "temperature": config.get("temperature"),
        "max_output_tokens": config.get("max_output_tokens"),
        "budgets": config.get("budgets"),
        "cache": config.get("cache"),
        "judge": config.get("judge"),
        "judge_config": config.get("judge_config"),
        "benchmark_version": config.get("benchmark_version"),
        "corpus_path": config.get("corpus_path"),
        "search_snapshot": config.get("search_snapshot"),
        "reflection": config.get("reflection"),
    }
    encoded = json.dumps(relevant, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _task_hash(task: dict[str, Any]) -> str:
    encoded = json.dumps(task, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _matches_current(
    result: dict[str, Any], agent_config: dict[str, Any], task: dict[str, Any], config: dict[str, Any]
) -> bool:
    return (
        result.get("config_hash") == _config_hash(agent_config, config)
        and result.get("task_hash") == _task_hash(task)
    )
