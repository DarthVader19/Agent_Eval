from __future__ import annotations

import time
from typing import Any

from agents.base import AgentStep, BaseAgent, Budget, estimate_tokens
from agents.runtime import execute_step, finish_trace
from tools.toolbox import ToolBox


class PlanExecuteAgent(BaseAgent):
    """Plan first, execute one action at a time, and replan from observed failures."""

    def run(
        self,
        task: dict[str, Any],
        toolbox: ToolBox | None = None,
        budget: Budget | dict[str, Any] | None = None,
    ):
        limits = self._budget(task, budget)
        tools = toolbox or ToolBox(cache_enabled=bool(self.config.get("cache_enabled", True)))
        started = time.monotonic()
        steps: list[AgentStep] = []
        context: list[str] = []
        plan_prompt = (
            f"Create a short ordered plan for this task, using only tools from "
            f"calculator, python, search, retrieve, read_file when needed.\nTask: {task['prompt']}\n"
            'Return JSON: {"plan":["step 1","step 2"]}. Do not answer the task yet.'
        )
        plan_output_limit = min(
            int(self.config.get("max_output_tokens", 512)),
            limits.max_tokens - estimate_tokens(plan_prompt),
        )
        if plan_output_limit < 1:
            return finish_trace("PlanExecute", task, steps, None, started, "budget_exhausted")
        original_timeout = getattr(self.llm, "timeout", None)
        if original_timeout is not None:
            setattr(self.llm, "timeout", min(float(original_timeout), limits.max_seconds))
        try:
            plan_response = self._complete(plan_prompt, max_output_tokens=plan_output_limit)
            plan_value = self._json_response(plan_response.text)
            plan_items = plan_value.get("plan", [])
            planning = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(plan_items))
            steps.append(
                AgentStep(
                    step_id="plan",
                    step_type="plan",
                    input_tokens=plan_response.input_tokens,
                    output_tokens=plan_response.output_tokens,
                    timestamp=time.time(),
                    action={"plan": plan_items},
                    observation=plan_response.text,
                    metadata={"cached": plan_response.cached},
                )
            )
        except Exception as exc:
            steps.append(AgentStep(step_id="plan-error", step_type="plan", timestamp=time.time(), error=str(exc)))
            return finish_trace("PlanExecute", task, steps, None, started, "model_error")
        finally:
            if original_timeout is not None:
                setattr(self.llm, "timeout", original_timeout)

        if sum(step.input_tokens + step.output_tokens for step in steps) > limits.max_tokens:
            return finish_trace("PlanExecute", task, steps, None, started, "budget_exhausted")
        context.append(f"Initial plan: {planning}")
        answer = None
        failure = None
        for _ in range(max(0, limits.max_steps - 1)):
            answer, failure = execute_step(
                self, task, tools, limits, steps, context, started, planning=planning
            )
            if answer is not None or failure is not None:
                break
            if any(step.error for step in steps[-2:]):
                context.append("Replan based on the failed tool and successful steps so far.")
        if answer is None and failure is None:
            failure = "budget_exhausted"
        return finish_trace("PlanExecute", task, steps, answer, started, failure)
