from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from benchmarks.task_generator import REQUIRED_FIELDS

KNOWN_TOOLS = {"calculator", "python", "search", "retrieve", "read_file"}


def validate_tasks(tasks: list[dict[str, Any]], *, expected_count: int | None = None) -> dict[str, Any]:
    errors: list[str] = []
    ids = [task.get("id") for task in tasks]
    duplicates = [task_id for task_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate ids: {duplicates[:5]}")
    if expected_count is not None and len(tasks) != expected_count:
        errors.append(f"Expected {expected_count} tasks, got {len(tasks)}.")
    for index, task in enumerate(tasks):
        missing = REQUIRED_FIELDS - task.keys()
        if missing:
            errors.append(f"Task at row {index + 1} lacks {sorted(missing)}.")
        if not str(task.get("prompt", "")).strip() or not str(task.get("expected_answer", "")).strip():
            errors.append(f"Task {task.get('id')} has an empty prompt or ground truth.")
        if task.get("split") not in {"dev", "test"}:
            errors.append(f"Task {task.get('id')} has an invalid split.")
        if int(task.get("max_steps", 0)) < 1:
            errors.append(f"Task {task.get('id')} has no usable step budget.")
        if not isinstance(task.get("required_tools"), list):
            errors.append(f"Task {task.get('id')} required_tools must be a list.")
        elif set(task["required_tools"]) - KNOWN_TOOLS:
            unknown_tools = sorted(set(task["required_tools"]) - KNOWN_TOOLS)
            errors.append(f"Task {task.get('id')} requires unknown tools: {unknown_tools}.")
        if task.get("grader_type") not in {"normalized_exact", "numeric_tolerance", "fuzzy_match", "llm_judge"}:
            errors.append(f"Task {task.get('id')} has an unsupported grader_type.")
        if "[specific query" in str(task.get("prompt", "")) or "expected_output_here" in str(task.get("expected_answer", "")):
            errors.append(f"Task {task.get('id')} still contains a placeholder.")
    test_count = sum(task.get("split") == "test" for task in tasks)
    test_fraction = test_count / len(tasks) if tasks else 0.0
    return {
        "valid": not errors,
        "count": len(tasks),
        "fault_cases": sum(bool(task.get("fault_config")) for task in tasks),
        "test_fraction": test_fraction,
        "category_counts": dict(Counter(task.get("category", "unknown") for task in tasks)),
        "errors": errors,
    }


def validate_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        tasks = [json.loads(line) for line in file if line.strip()]
    result = validate_tasks(tasks)
    name = Path(path).name
    if name == "custom_tasks.jsonl":
        if len(tasks) < 100:
            result["errors"].append("Custom suite must include at least 100 tasks.")
        if result["fault_cases"] < 20:
            result["errors"].append("Custom suite must include at least 20 fault-injection tasks.")
        if not 0.18 <= result["test_fraction"] <= 0.22:
            result["errors"].append("Custom test split must be approximately 20%.")
    if name == "gaia_style_tasks.jsonl" and not 30 <= len(tasks) <= 50:
        result["errors"].append("GAIA-style suite must contain 30-50 tasks.")
    result["valid"] = not result["errors"]
    return result


if __name__ == "__main__":
    import json
    import sys

    for filename in sys.argv[1:] or ["benchmarks/custom_tasks.jsonl", "benchmarks/gaia_style_tasks.jsonl"]:
        result = validate_file(filename)
        print(filename, json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
