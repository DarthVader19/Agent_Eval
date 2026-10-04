from __future__ import annotations

import pytest

from agents.base import Budget, ModelResponse
from agents.direct_baseline import DirectBaselineAgent
from harness.judge import LLMJudge
from harness.reflection import ReflectionLoop
from harness.runner import EvaluationHarness, _include_judge_usage
from harness.stats import paired_binary_test, paired_bootstrap_ci
from tools.calculator import calculate
from tools.fault_injection import FaultInjector
from tools.python_sandbox import run_python
from tools.toolbox import ToolBox


class SequenceModel:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.last_options: tuple[str, float, int] | None = None

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0, max_output_tokens: int = 512) -> ModelResponse:
        self.prompts.append(prompt)
        self.last_options = (system, temperature, max_output_tokens)
        if not self.responses:
            raise AssertionError("Unexpected model call.")
        return ModelResponse(self.responses.pop(0), 12, 4)


def _task(expected: str = "42") -> dict:
    return {
        "id": "runner-test",
        "prompt": "Answer the test question.",
        "expected_answer": expected,
        "category": "test",
        "difficulty": "easy",
        "required_tools": [],
        "max_steps": 3,
        "grader_type": "normalized_exact",
        "rubric": {"type": "normalized_exact"},
        "split": "dev",
        "fault_config": {},
    }


def test_normalized_and_fuzzy_graders_are_deterministic():
    judge = LLMJudge({})
    exact = judge.evaluate(_task("Hello World"), " hello   world! ")
    fuzzy = judge.evaluate(
        {**_task("hello world"), "grader_type": "fuzzy_match", "rubric": {"type": "fuzzy_match", "threshold": 0.5}},
        "hello world!",
    )
    assert exact["passed"]
    assert fuzzy["score"] == 1.0


def test_numeric_tolerance_grader():
    judge = LLMJudge({})
    task = {
        **_task("3.14159"),
        "grader_type": "numeric_tolerance",
        "rubric": {"type": "numeric_tolerance", "tolerance": 0.001},
    }
    assert judge.evaluate(task, "3.1416")["passed"]
    assert not judge.evaluate(task, "3.2")["passed"]


def test_llm_judge_requires_minimum_calibration_sample():
    result = LLMJudge({}).calibrate([])
    assert not result["calibration_complete"]
    assert result["num_samples"] == 0


def test_llm_judge_calibration_accounts_for_token_usage():
    class JudgeModel:
        def complete(self, _prompt, *, system="", temperature=0.0, max_output_tokens=512):
            return ModelResponse('{"score":1,"reason":"correct"}', 12, 4)

    judge = LLMJudge({"votes": 1}, model_client=JudgeModel())
    sample = {
        "task": {"prompt": "test", "expected_answer": "42", "rubric": {"threshold": 0.7}},
        "answer": "42",
        "label": 1,
    }

    report = judge.calibrate([sample] * 30)
    metrics = {
        "tokens_in": 0,
        "tokens_out": 0,
        "total_tokens": 0,
        "billed_tokens_in": 0,
        "billed_tokens_out": 0,
        "judge_tokens_in": 0,
        "judge_tokens_out": 0,
    }
    _include_judge_usage(metrics, report)

    assert report["passed"]
    assert report["tokens_in"] == 360
    assert report["tokens_out"] == 120
    assert metrics["total_tokens"] == 480
    assert metrics["billed_tokens_in"] == 360
    assert metrics["billed_tokens_out"] == 120


def test_safe_calculator_and_python_sandbox():
    assert calculate("2 + 3 * 4") == 14


def test_sandbox_executes_restricted_python_and_rejects_imports():
    assert run_python("print(sum([2, 3, 4]))") == "9"
    with pytest.raises(ValueError, match="Unsupported Python syntax"):
        run_python("import os\nprint(os.getcwd())")


def test_tool_fault_injection_is_repeatable():
    injector = FaultInjector({"calculator": {"type": "transient"}})
    with pytest.raises(RuntimeError, match="transient"):
        injector.apply("calculator", "4")
    assert injector.apply("calculator", "4") == "4"


def test_toolbox_logs_and_faults_io(tmp_path):
    toolbox = ToolBox(
        cache_enabled=False,
        cache_path=tmp_path / "tools.sqlite3",
        fault_config={"calculator": {"type": "wrong"}},
    )
    assert toolbox.invoke("calculator", {"expression": "2+2"}) == "Injected plausible but incorrect result."
    assert toolbox.call_log[0]["input"] == {"expression": "2+2"}
    assert toolbox.call_log[0]["output"] == "Injected plausible but incorrect result."


def test_toolbox_caches_successful_results(tmp_path):
    toolbox = ToolBox(cache_enabled=True, cache_path=tmp_path / "tools.sqlite3")
    assert toolbox.invoke("calculator", {"expression": "2+2"}) == "4"
    assert toolbox.invoke("calculator", {"expression": "2+2"}) == "4"
    assert toolbox.call_log[-1]["cached"]


def test_toolbox_rejects_disabled_tools():
    toolbox = ToolBox(cache_enabled=False, allowed_tools=[])
    with pytest.raises(ValueError, match="disabled by this experiment"):
        toolbox.invoke("calculator", {"expression": "2+2"})


def test_cost_cap_blocks_run_before_model_call(tmp_path):
    harness = EvaluationHarness(
        {
            "model": "priced-model",
            "spend_cap_usd": 1.0,
            "pricing_per_million_tokens": {"priced-model": [1000, 1000]},
            "budgets": {"max_tokens": 4000},
            "results_dir": tmp_path / "results",
        }
    )
    with pytest.raises(RuntimeError, match="exceeds configured spend cap"):
        harness.run_evaluation([{"type": "direct"}], [_task()], repeats=1)


def test_runs_repeat_and_resume_from_sqlite(tmp_path):
    model = SequenceModel("42", "42")
    harness = EvaluationHarness(
        {
            "model": "test",
            "cache": {"enabled": False},
            "budgets": {"max_steps": 1, "max_tokens": 1000, "max_seconds": 5},
            "results_dir": tmp_path / "run",
            "llm_client": model,
        }
    )
    # The fake client is passed on each agent config just as a provider client can be injected by a test.
    configs = [{"type": "direct", "llm_client": model}]
    result = harness.run_evaluation(configs, [_task()], repeats=2)
    assert len(result["detailed_results"]) == 2
    assert result["summary"]["direct_off"]["success_rate"] == 1.0
    assert (tmp_path / "run" / "results.sqlite3").exists()
    resumed = harness.run_evaluation(configs, [_task()], repeats=2)
    assert len(resumed["detailed_results"]) == 2
    assert model.responses == []


def test_harness_persists_paired_reflection_metrics(tmp_path):
    model = SequenceModel("wrong", '{"critique":"check the answer","revised_approach":"answer accurately"}', "42")
    harness = EvaluationHarness(
        {
            "model": "test",
            "cache": {"enabled": False},
            "budgets": {"max_steps": 2, "max_tokens": 1000, "max_seconds": 5},
            "reflection": {"enabled": True, "max_cycles": 1},
            "results_dir": tmp_path / "reflection-run",
        }
    )
    result = harness.run_evaluation(
        [{"type": "direct", "llm_client": model}],
        [_task()],
        repeats=1,
    )
    assert {row["setting"] for row in result["detailed_results"]} == {"off", "on"}
    assert result["summary"]["reflection"]["direct"]["rescue_rate"] == 1.0
    assert result["summary"]["reflection"]["direct"]["regression_rate"] == 0.0


def test_reflection_omits_ground_truth_from_critic_and_revision():
    model = SequenceModel("wrong", '{"critique":"recheck arithmetic","revised_approach":"calculate carefully"}', "42")
    agent = DirectBaselineAgent({"llm_client": model})
    task = _task("42")
    original = agent.run(task)
    result = ReflectionLoop({"max_reflections": 9}).reflect(
        agent,
        task,
        original,
        toolbox=ToolBox(cache_enabled=False),
        budget=Budget(max_steps=3, max_tokens=1000, max_seconds=10),
        evaluator=lambda answer: LLMJudge({}).evaluate(task, answer),
    )
    assert result.reflected_answer == "42"
    assert result.reflected_score == 1.0
    assert "42" not in model.prompts[1]
    assert len(result.reflection_steps) == 1


def test_paired_statistics():
    delta, low, high = paired_bootstrap_ci([0, 1, 0], [1, 1, 0], iterations=500, seed=2)
    test = paired_binary_test([False, True], [True, False])
    assert delta == pytest.approx(1 / 3)
    assert low <= delta <= high
    assert test["discordant_left"] == 1
    assert test["discordant_right"] == 1


def test_pilot_usage_drives_cost_estimate(tmp_path):
    harness = EvaluationHarness(
        {
            "model": "priced-model",
            "pricing_per_million_tokens": {"priced-model": [1.0, 2.0]},
            "budgets": {"max_tokens": 4000},
            "pilot_usage": {"input_tokens": 100, "output_tokens": 20, "safety_uplift": 1.25},
            "results_dir": tmp_path / "estimate",
        }
    )
    estimate = harness.estimate_spend([{"type": "direct", "model": "priced-model"}], [_task()], 1)
    assert estimate["basis"] == "pilot_mean_with_safety_uplift"
    assert estimate["assumed_input_tokens_per_run"] == 125
    assert estimate["assumed_output_tokens_per_run"] == 25
