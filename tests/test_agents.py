from __future__ import annotations

from agents.base import Budget, ModelResponse
from agents.direct_baseline import DirectBaselineAgent
from agents.langgraph_agent import LangGraphAgent
from agents.plan_execute_agent import PlanExecuteAgent
from agents.react_agent import ReActAgent
from benchmarks.task_generator import BenchmarkGenerator
from benchmarks.validate import validate_tasks
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
            raise AssertionError("Unexpected additional model call.")
        return ModelResponse(self.responses.pop(0), 20, 5)


def _task() -> dict:
    return {
        "id": "agent-test",
        "prompt": "What is 2 + 2?",
        "expected_answer": "4",
        "category": "math",
        "difficulty": "easy",
        "required_tools": ["calculator"],
        "max_steps": 5,
        "grader_type": "numeric_tolerance",
        "rubric": {"type": "numeric_tolerance", "tolerance": 0},
        "split": "dev",
        "fault_config": {},
    }


def test_react_uses_shared_toolbox_and_records_io(tmp_path):
    model = SequenceModel(
        '{"type":"tool","tool":"calculator","params":{"expression":"2+2"}}',
        '{"type":"final","answer":"4"}',
    )
    agent = ReActAgent({"llm_client": model, "max_output_tokens": 20})
    toolbox = ToolBox(cache_enabled=False, cache_path=tmp_path / "tools.sqlite3")

    trace = agent.run(_task(), toolbox, Budget(max_steps=3, max_tokens=1000, max_seconds=10))

    assert trace.final_answer == "4"
    assert trace.success
    tool_step = next(step for step in trace.steps if step.step_type == "tool")
    assert tool_step.action == {"tool": "calculator", "params": {"expression": "2+2"}}
    assert tool_step.observation == "4"
    assert trace.total_tokens == 50
    assert model.last_options is not None
    assert model.last_options[1] == 0.0


def test_plan_execute_creates_plan_and_replans_after_tool_failure(tmp_path):
    model = SequenceModel(
        '{"plan":["calculate 2 + 2"]}',
        '{"type":"tool","tool":"calculator","params":{"expression":"2+2"}}',
        '{"type":"final","answer":"4"}',
    )
    agent = PlanExecuteAgent({"llm_client": model, "max_output_tokens": 20})
    toolbox = ToolBox(
        cache_enabled=False,
        cache_path=tmp_path / "tools.sqlite3",
        fault_config={"calculator": {"type": "transient"}},
    )

    trace = agent.run(_task(), toolbox, Budget(max_steps=4, max_tokens=1000, max_seconds=10))

    assert trace.final_answer == "4"
    assert len(trace.steps) >= 4
    assert trace.errors
    assert trace.total_tokens == 75


def test_langgraph_agent_uses_real_graph_and_checkpointed_state(tmp_path):
    model = SequenceModel('{"type":"final","answer":"4"}')
    agent = LangGraphAgent({"llm_client": model})
    trace = agent.run(_task(), ToolBox(cache_enabled=False, cache_path=tmp_path / "tools.sqlite3"))
    assert trace.final_answer == "4"
    assert trace.agent_type == "LangGraph"
    assert trace.steps[0].step_type == "reason"


def test_direct_baseline_does_not_call_tools():
    model = SequenceModel("4")
    trace = DirectBaselineAgent({"llm_client": model}).run(_task())
    assert trace.final_answer == "4"
    assert not any(step.step_type == "tool" for step in trace.steps)


def test_budget_exhaustion_is_explicit():
    model = SequenceModel('{"type":"tool","tool":"calculator","params":{"expression":"2+2"}}')
    trace = ReActAgent({"llm_client": model}).run(
        _task(), ToolBox(cache_enabled=False), Budget(max_steps=1, max_tokens=1000, max_seconds=10)
    )
    assert not trace.success
    assert trace.failure_reason == "budget_exhausted"


def test_frozen_task_suite_has_required_counts_and_fields():
    generator = BenchmarkGenerator()
    custom = generator.generate_custom_tasks()
    gaia = generator.generate_gaia_style_tasks()
    custom_report = validate_tasks(custom)
    gaia_report = validate_tasks(gaia)
    assert custom_report["count"] == 130
    assert custom_report["fault_cases"] >= 20
    assert custom_report["test_fraction"] == 0.2
    assert gaia_report["count"] == 40
    assert all(set(task) >= {"id", "prompt", "expected_answer", "max_steps", "grader_type", "split", "fault_config"} for task in custom + gaia)
