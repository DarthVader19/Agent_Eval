from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any

from harness.stats import bootstrap_mean_ci


class MetricsCalculator:
    metric_names = ["success_rate", "token_efficiency", "error_recovery_rate", "latency"]

    def calculate_task_metrics(self, trace: Any, score: dict[str, Any], _task: dict[str, Any]) -> dict[str, Any]:
        passed = bool(score["passed"])
        error_steps = [step for step in trace.steps if step.error]
        duration_steps = [step.duration for step in trace.steps]
        cached_steps = [step for step in trace.steps if step.metadata.get("cached")]
        uncached_times = [step.duration for step in trace.steps if not step.metadata.get("cached")]
        model_steps = [step for step in trace.steps if step.step_type in {"reason", "model", "final", "reflection"}]
        billed_steps = [step for step in model_steps if not step.metadata.get("cached")]
        return {
            "success_rate": float(passed),
            "token_efficiency": trace.total_tokens if passed else None,
            "tokens_per_step": trace.total_tokens / max(1, len(trace.steps)),
            "error_recovery_rate": float(passed) if error_steps else None,
            "has_step_error": bool(error_steps),
            "total_latency": trace.total_time,
            "uncached_latency": sum(uncached_times),
            "avg_step_latency": mean(duration_steps) if duration_steps else 0.0,
            "total_tokens": trace.total_tokens,
            "tokens_in": sum(step.input_tokens for step in trace.steps),
            "tokens_out": sum(step.output_tokens for step in trace.steps),
            "billed_tokens_in": sum(step.input_tokens for step in billed_steps),
            "billed_tokens_out": sum(step.output_tokens for step in billed_steps),
            "judge_tokens_in": 0,
            "judge_tokens_out": 0,
            "num_steps": len(trace.steps),
            "num_errors": len(trace.errors),
            "cached_calls": len(cached_steps),
            "score": float(score["score"]),
            "grader": score["method"],
        }

    def calculate_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            agent = result.get("agent_type", "unknown")
            setting = result.get("setting")
            key = f"{agent}_{setting}" if setting in {"on", "off"} else agent
            grouped[key].append(result)
        summary = {key: self._summarize(rows) for key, rows in sorted(grouped.items())}
        summary["overall"] = self._summarize(results)
        return summary

    def _summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {"total_runs": 0, "success_rate": 0.0}
        success = [float(item.get("success", False)) for item in results]
        tokens = [item["metrics"]["total_tokens"] for item in results if "metrics" in item]
        latency = [item["metrics"]["total_latency"] for item in results if "metrics" in item]
        successful_tokens = [
            item["metrics"]["total_tokens"] for item in results
            if item.get("success") and "metrics" in item
        ]
        recovered = [
            item["metrics"]["error_recovery_rate"] for item in results
            if item.get("metrics", {}).get("error_recovery_rate") is not None
        ]
        lo, hi = bootstrap_mean_ci(success, seed=17)
        task_groups: dict[str, list[bool]] = defaultdict(list)
        for item in results:
            task_groups[str(item.get("task_id"))].append(bool(item.get("success")))
        return {
            "total_runs": len(results),
            "successful_runs": int(sum(success)),
            "success_rate": mean(success),
            "pass_at_1": mean(success),
            "success_rate_ci95": [lo, hi],
            "all_repeats_correct_rate": mean(
                [float(all(values)) for values in task_groups.values()]
            ),
            "average_tokens": mean(tokens) if tokens else 0.0,
            "average_tokens_in": mean([
                item["metrics"]["tokens_in"] for item in results if "metrics" in item
            ]),
            "average_tokens_out": mean([
                item["metrics"]["tokens_out"] for item in results if "metrics" in item
            ]),
            "tokens_per_success": mean(successful_tokens) if successful_tokens else None,
            "cost_per_success_usd": _mean_or_none([
                float(item.get("cost_usd", 0.0)) for item in results if item.get("success")
            ]),
            "average_latency_seconds": mean(latency) if latency else 0.0,
            "p50_latency_seconds": median(latency) if latency else 0.0,
            "p95_latency_seconds": _percentile(latency, 0.95),
            "error_recovery_rate": mean(recovered) if recovered else None,
            "natural_fault_error_recovery_rate": _recovery_for_faults(results, injected=False),
            "injected_fault_error_recovery_rate": _recovery_for_faults(results, injected=True),
            "error_runs": sum(bool(item.get("metrics", {}).get("has_step_error")) for item in results),
            "by_category": self._breakdown(results, "task_category"),
            "by_difficulty": self._breakdown(results, "task_difficulty"),
            "by_fault": self._breakdown(results, "fault_type"),
        }

    def _breakdown(self, results: list[dict[str, Any]], key: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in results:
            value = item.get(key) or "none"
            buckets[str(value)].append(item)
        return {
            name: {
                "count": len(items),
                "success_rate": mean(float(item.get("success", False)) for item in items),
            }
            for name, items in sorted(buckets.items())
        }

    def generate_comparison_table(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "agent_type": name,
                "success_rate": values.get("success_rate", 0),
                "success_rate_ci95": values.get("success_rate_ci95"),
                "average_tokens": values.get("average_tokens"),
                "tokens_per_success": values.get("tokens_per_success"),
                "average_latency_seconds": values.get("average_latency_seconds"),
                "p95_latency_seconds": values.get("p95_latency_seconds"),
            }
            for name, values in summary.items() if name != "overall"
        ]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _recovery_for_faults(results: list[dict[str, Any]], *, injected: bool) -> float | None:
    values = [
        item["metrics"]["error_recovery_rate"]
        for item in results
        if bool(item.get("fault_type") not in {None, "", "none"}) is injected
        and item.get("metrics", {}).get("error_recovery_rate") is not None
    ]
    return mean(values) if values else None
