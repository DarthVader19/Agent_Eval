from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from agents.base import AgentStep, Budget
from tools.toolbox import ToolBox


@dataclass
class ReflectionResult:
    original_answer: str
    original_score: float
    reflected_answer: str
    reflected_score: float
    improvement: float
    reflection_steps: list[AgentStep]
    tokens_used: int
    time_spent: float
    trace: Any
    score: dict[str, Any]


class ReflectionLoop:
    """Critique and retry at most twice; ground truth is never given to the agent."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_reflections = min(2, max(0, int(config.get("max_reflections", config.get("max_cycles", 2)))))

    def reflect(
        self,
        agent: Any,
        task: dict[str, Any],
        original_trace: Any,
        *,
        toolbox: ToolBox,
        budget: Budget,
        evaluator: Callable[[str], dict[str, Any]],
    ) -> ReflectionResult:
        started = time.monotonic()
        original_score = evaluator(str(original_trace.final_answer))
        current_trace = original_trace
        current_score = original_score
        reflections: list[AgentStep] = []
        attempt_steps: list[AgentStep] = []
        used_tokens = int(original_trace.total_tokens)
        used_seconds = float(original_trace.total_time)
        prompt = str(task["prompt"])

        for cycle in range(self.max_reflections):
            if current_score["passed"]:
                break
            remaining_tokens = max(0, budget.max_tokens - used_tokens)
            remaining_seconds = max(0.0, budget.max_seconds - used_seconds)
            if remaining_tokens <= 0 or remaining_seconds <= 0:
                break
            critique_prompt = (
                "Review this attempt and suggest one concrete correction. You are not given the reference "
                "answer. Return JSON with a short critique and a revised approach.\n"
                f"Task: {prompt}\nPrevious response: {current_trace.final_answer}\n"
                f"Trace: {[{'type': step.step_type, 'action': step.action, 'result': step.observation, 'error': step.error} for step in current_trace.steps]}\n"
                f"Cycle: {cycle + 1}"
            )
            critique_token_budget = max(1, remaining_tokens - len(critique_prompt) // 4)
            original_timeout = getattr(agent.llm, "timeout", None)
            if original_timeout is not None:
                setattr(agent.llm, "timeout", min(float(original_timeout), remaining_seconds))
            try:
                response = agent._complete(
                    critique_prompt,
                    max_output_tokens=min(int(agent.config.get("max_output_tokens", 512)), critique_token_budget),
                )
            except Exception as exc:
                reflections.append(
                    AgentStep(step_id=f"reflection-error-{cycle + 1}", step_type="reflection", error=str(exc))
                )
                break
            finally:
                if original_timeout is not None:
                    setattr(agent.llm, "timeout", original_timeout)
            critique = agent._json_response(response.text)
            reflection_step = AgentStep(
                step_id=f"reflection-{cycle + 1}",
                step_type="reflection",
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                timestamp=time.time(),
                observation=response.text,
                metadata={"cycle": cycle + 1, "cached": response.cached},
            )
            reflections.append(reflection_step)
            used_tokens += response.input_tokens + response.output_tokens
            if used_tokens > budget.max_tokens:
                break
            revised_task = {
                key: value for key, value in task.items()
                if key not in {"expected_answer", "rubric", "reference_answer", "label"}
            }
            revised_task["prompt"] = (
                f"{prompt}\n\nA previous attempt was unsuccessful. Critique: "
                f"{critique.get('critique', '')}\nRevised approach: {critique.get('revised_approach', '')}"
            )
            remaining = Budget(
                max_steps=budget.max_steps,
                max_tokens=max(1, budget.max_tokens - used_tokens),
                max_seconds=max(0.1, budget.max_seconds - used_seconds),
            )
            revised_trace = agent.run(revised_task, toolbox, remaining)
            attempt_steps.extend(revised_trace.steps)
            used_tokens += revised_trace.total_tokens
            used_seconds += revised_trace.total_time
            score = evaluator(str(revised_trace.final_answer))
            if score["score"] > current_score["score"]:
                current_trace = revised_trace
                current_score = score

        current_trace.reflection_used = bool(reflections)
        current_trace.reflection_steps = reflections
        current_trace.total_tokens = used_tokens
        current_trace.total_time = used_seconds
        current_trace.steps = [*original_trace.steps, *reflections, *attempt_steps]
        return ReflectionResult(
            original_answer=str(original_trace.final_answer),
            original_score=float(original_score["score"]),
            reflected_answer=str(current_trace.final_answer),
            reflected_score=float(current_score["score"]),
            improvement=float(current_score["score"]) - float(original_score["score"]),
            reflection_steps=reflections,
            tokens_used=used_tokens - int(original_trace.total_tokens),
            time_spent=time.monotonic() - started,
            trace=current_trace,
            score=current_score,
        )
