#!/usr/bin/env python3
"""Config-driven command-line entry point for the evaluation framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from benchmarks.task_generator import freeze_benchmarks
from benchmarks.validate import validate_file
from harness.runner import AGENT_TYPES, EvaluationHarness


ROOT = Path(__file__).resolve().parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Experiment config does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must contain a YAML mapping: {path}")
    return config


def load_tasks(benchmark_path: str | Path) -> list[dict[str, Any]]:
    path = Path(benchmark_path)
    if not path.is_absolute():
        path = ROOT / path
    validation = validate_file(path)
    if not validation["valid"]:
        raise ValueError(f"Benchmark validation failed for {path}: {validation['errors']}")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible agent evaluation and benchmarking")
    parser.add_argument("--agent", choices=[*AGENT_TYPES, "all"], default="all")
    parser.add_argument("--benchmark", choices=["custom", "gaia", "gaia_style", "all"], default="custom")
    parser.add_argument("--config", default="configs/experiments/pilot.yaml")
    parser.add_argument("--model-role", choices=["development", "primary"])
    parser.add_argument("--pilot-results", help="Pilot summary.json used to estimate a full run from observed token use.")
    parser.add_argument("--split", choices=["dev", "test", "all"])
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--reflection", action="store_true")
    parser.add_argument("--framework-compare", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", help="Reuse the result directory name from a prior run.")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--generate-benchmarks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_dotenv(ROOT / ".env", override=False)
    if args.generate_benchmarks:
        print(json.dumps(freeze_benchmarks(ROOT / "benchmarks"), indent=2))
        return 0
    if args.task_limit is not None and args.task_limit < 1:
        raise ValueError("--task-limit must be positive.")
    config = load_config(args.config)
    if args.pilot_results:
        pilot_path = Path(args.pilot_results)
        if not pilot_path.is_absolute():
            pilot_path = ROOT / pilot_path
        pilot_summary = json.loads(pilot_path.read_text(encoding="utf-8"))
        agent_summaries = [
            value for key, value in pilot_summary.items()
            if key not in {"overall", "run", "reflection"} and value.get("total_runs", 0)
        ]
        if not agent_summaries:
            raise ValueError(f"Pilot summary contains no per-agent token usage: {pilot_path}")
        config["pilot_usage"] = {
            "input_tokens": max(float(value["average_tokens_in"]) for value in agent_summaries),
            "output_tokens": max(float(value["average_tokens_out"]) for value in agent_summaries),
            "safety_uplift": 1.25,
            "source": str(pilot_path),
        }
    if config.get("name") == "full" and not args.estimate_only and not config.get("pilot_usage"):
        raise ValueError("Run the pilot first and pass --pilot-results <pilot summary.json> before starting a full run.")
    config["resume_run"] = args.resume or config.get("resume_run")
    if config["resume_run"]:
        config["results_dir"] = str(ROOT / "results" / config["resume_run"])
    else:
        config.pop("results_dir", None)
    if args.reflection:
        config["reflection"] = {**config.get("reflection", {}), "enabled": True}
    if "cache" not in config:
        config["cache"] = {"enabled": True}

    benchmark_paths: list[Path] = []
    if args.benchmark in {"custom", "all"}:
        benchmark_paths.append(ROOT / "benchmarks" / "custom_tasks.jsonl")
    if args.benchmark in {"gaia", "gaia_style", "all"}:
        benchmark_paths.append(ROOT / "benchmarks" / "gaia_style_tasks.jsonl")
    tasks = [task for path in benchmark_paths for task in load_tasks(path)]
    selected_split = args.split or config.get("split", "dev")
    if isinstance(selected_split, list):
        selected_split = "all"
    if selected_split != "all":
        tasks = [task for task in tasks if task["split"] == selected_split]
    if args.task_limit is not None:
        tasks = tasks[:args.task_limit]
    elif config.get("task_limit"):
        tasks = tasks[: int(config["task_limit"])]
    if not tasks:
        raise ValueError("No benchmark tasks remain after applying benchmark/split/limit filters.")

    configured_agents = config.get("agents", ["react", "plan_execute", "langgraph", "direct"])
    if args.framework_compare:
        selected_agents = ["langgraph", "crewai", "autogen"]
    else:
        selected_agents = list(AGENT_TYPES) if args.agent == "all" else [args.agent]
        selected_agents = [name for name in selected_agents if name in configured_agents]
    if not selected_agents:
        raise ValueError(f"Agent selection {args.agent!r} is not enabled by the config.")
    model_role = args.model_role or str(config.get("default_model_role", "development"))
    model = str(config.get(f"{model_role}_model", config.get("model", "gpt-4o-mini")))
    config["model"] = model
    agent_configs = [
        {
            "type": name,
            "model": model,
            "temperature": float(config.get("temperature", 0.0)),
            "max_output_tokens": int(config.get("max_output_tokens", 512)),
        }
        for name in selected_agents
    ]
    if not args.estimate_only:
        # Keys are read directly from the process environment.
        import os

        if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            raise RuntimeError("No API key found. Set OPENAI_API_KEY or configure an OpenAI-compatible provider.")

    harness = EvaluationHarness(config)
    result = harness.run_evaluation(
        agent_configs,
        tasks,
        parallel=args.parallel,
        max_workers=args.max_workers,
        repeats=args.repeats,
        estimate_only=args.estimate_only,
    )
    print(json.dumps(result["estimate"], indent=2))
    if args.estimate_only:
        return 0
    print(f"Completed {len(result['detailed_results'])} recorded runs.")
    for name, values in result["summary"].items():
        if name in {"overall", "run", "reflection"}:
            continue
        print(
            f"{name}: success={values['success_rate']:.1%} "
            f"95% CI={values.get('success_rate_ci95')} "
            f"tokens/success={values.get('tokens_per_success')} "
            f"latency-p50={values.get('p50_latency_seconds', 0):.3f}s"
        )
    print(f"Results: {harness.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
