from __future__ import annotations

import re
from typing import Any

from harness.model import OpenAICompatibleModel


class LLMJudge:
    """Deterministic graders first; calibrated, separate-model judging for free-form answers."""

    def __init__(self, config: dict[str, Any] | None = None, model_client: Any = None):
        self.config = config or {}
        self.model = str(self.config.get("model", "gpt-4o"))
        self.temperature = 0.0
        self.votes = max(1, int(self.config.get("votes", 3)))
        self.model_client = model_client or OpenAICompatibleModel(
            self.model,
            cache_path=self.config.get("cache_path", "results/cache.sqlite3"),
            cache_enabled=bool(self.config.get("cache_enabled", True)),
        )

    def evaluate(self, task: dict[str, Any], answer: str) -> dict[str, Any]:
        grader = task.get("grader_type") or task.get("rubric", {}).get("type", "normalized_exact")
        if grader in {"exact", "exact_match", "normalized_exact"}:
            return self._exact(task, answer)
        if grader in {"fuzzy", "fuzzy_match"}:
            return self._fuzzy(task, answer)
        if grader in {"numeric", "numeric_tolerance"}:
            return self._numeric(task, answer)
        if grader in {"llm", "llm_judge", "rubric"}:
            return self._llm_judge(task, answer)
        raise ValueError(f"Unsupported grader type: {grader}")

    def _exact(self, task: dict[str, Any], answer: str) -> dict[str, Any]:
        expected = _normalize(str(task.get("expected_answer", "")))
        actual = _normalize(str(answer))
        passed = bool(expected) and expected == actual
        return {"score": float(passed), "passed": passed, "method": "normalized_exact", "confidence": 1.0}

    def _fuzzy(self, task: dict[str, Any], answer: str) -> dict[str, Any]:
        expected = _normalize(str(task.get("expected_answer", "")))
        actual = _normalize(str(answer))
        expected_tokens = set(expected.split())
        actual_tokens = set(actual.split())
        overlap = len(expected_tokens & actual_tokens) / len(expected_tokens) if expected_tokens else 0.0
        threshold = float(task.get("rubric", {}).get("threshold", 0.8))
        passed = overlap >= threshold
        return {"score": overlap, "passed": passed, "method": "fuzzy_match", "confidence": overlap}

    def _numeric(self, task: dict[str, Any], answer: str) -> dict[str, Any]:
        expected = _number(str(task.get("expected_answer", "")))
        actual = _number(answer)
        tolerance = float(task.get("rubric", {}).get("tolerance", 1e-6))
        if expected is None or actual is None:
            return {"score": 0.0, "passed": False, "method": "numeric_tolerance", "confidence": 0.0}
        passed = abs(expected - actual) <= tolerance
        return {
            "score": float(passed),
            "passed": passed,
            "method": "numeric_tolerance",
            "expected": expected,
            "actual": actual,
            "tolerance": tolerance,
            "confidence": 1.0,
        }

    def _llm_judge(self, task: dict[str, Any], answer: str) -> dict[str, Any]:
        criteria = task.get("rubric", {}).get("criteria", ["correctness", "completeness"])
        prompt = (
            "Evaluate the response against the task and rubric. The expected answer may be used only "
            "as reference. Return JSON with numeric score 0..1 and a brief reason.\n"
            f"Task: {task['prompt']}\nExpected answer/rubric: {task.get('expected_answer', task.get('rubric'))}\n"
            f"Response: {answer}\nCriteria: {criteria}"
        )
        scores: list[float] = []
        reasons: list[str] = []
        usage = {"tokens_in": 0, "tokens_out": 0, "billed_tokens_in": 0, "billed_tokens_out": 0}
        for _ in range(self.votes):
            result = self.model_client.complete(
                prompt,
                system="You are a strict, calibrated evaluator. Output valid JSON only.",
                temperature=0.0,
                max_output_tokens=256,
            )
            value = _parse_json(result.text)
            if not isinstance(value.get("score"), (int, float)):
                raise ValueError("LLM judge response did not contain a numeric score.")
            scores.append(max(0.0, min(1.0, float(value["score"]))))
            reasons.append(str(value.get("reason", "")))
            usage["tokens_in"] += result.input_tokens
            usage["tokens_out"] += result.output_tokens
            if not result.cached:
                usage["billed_tokens_in"] += result.input_tokens
                usage["billed_tokens_out"] += result.output_tokens
        score = sorted(scores)[len(scores) // 2]
        passed = score >= float(task.get("rubric", {}).get("threshold", 0.7))
        return {
            "score": score,
            "passed": passed,
            "method": "llm_judge_majority",
            "votes": scores,
            "reason": reasons[0],
            "confidence": 1.0 - (max(scores) - min(scores)),
            "judge_usage": usage,
        }

    def calibrate(self, labeled_samples: list[dict[str, Any]]) -> dict[str, Any]:
        if len(labeled_samples) < 30:
            return {
                "calibration_complete": False,
                "passed": False,
                "reason": "At least 30 hand-labeled samples are required.",
                "num_samples": len(labeled_samples),
            }
        predicted: list[int] = []
        observed: list[int] = []
        tokens_in = tokens_out = billed_tokens_in = billed_tokens_out = 0
        for sample in labeled_samples:
            result = self._llm_judge(sample["task"], sample["answer"])
            predicted.append(int(result["passed"]))
            observed.append(int(sample["label"]))
            usage = result.get("judge_usage", {})
            tokens_in += int(usage.get("tokens_in", 0))
            tokens_out += int(usage.get("tokens_out", 0))
            billed_tokens_in += int(usage.get("billed_tokens_in", 0))
            billed_tokens_out += int(usage.get("billed_tokens_out", 0))
        n = len(observed)
        observed_agreement = sum(a == b for a, b in zip(predicted, observed)) / n
        predicted_yes = sum(predicted) / n
        observed_yes = sum(observed) / n
        expected_agreement = predicted_yes * observed_yes + (1 - predicted_yes) * (1 - observed_yes)
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement) if expected_agreement < 1 else 1.0
        return {
            "calibration_complete": True,
            "passed": kappa >= float(self.config.get("minimum_kappa", 0.6)),
            "cohen_kappa": kappa,
            "agreement": observed_agreement,
            "num_samples": n,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "billed_tokens_in": billed_tokens_in,
            "billed_tokens_out": billed_tokens_out,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().strip().rstrip(".!?"))


def _number(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)", text.replace(",", ""))
    return float(match.group()) if match else None


def _parse_json(text: str) -> dict[str, Any]:
    import json

    start, end = text.find("{"), text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("LLM judge returned invalid JSON.")
    value = json.loads(text[start:end])
    if not isinstance(value, dict):
        raise ValueError("LLM judge returned a non-object JSON value.")
    return value
