from __future__ import annotations

import time
from typing import Any

from agents.base import AgentStep, BaseAgent, Budget, estimate_tokens
from agents.runtime import finish_trace


class DirectBaselineAgent(BaseAgent):
    """One-shot direct LLM baseline; no tools are exposed to the model."""

    def run(self, task: dict[str, Any], _toolbox: Any = None, budget: Budget | dict[str, Any] | None = None):
        limits = self._budget(task, budget)
        started = time.monotonic()
        prompt = f"Answer this task directly without tools:\n{task['prompt']}"
        output_limit = min(
            int(self.config.get("max_output_tokens", 512)),
            limits.max_tokens - estimate_tokens(prompt),
        )
        if output_limit < 1 or limits.max_seconds <= 0:
            return finish_trace("Direct", task, [], None, started, "budget_exhausted")
        original_timeout = getattr(self.llm, "timeout", None)
        if original_timeout is not None:
            setattr(self.llm, "timeout", min(float(original_timeout), limits.max_seconds))
        try:
            response = self._complete(prompt, max_output_tokens=output_limit)
        except Exception as exc:
            step = AgentStep(step_id="direct-error", step_type="model", timestamp=time.time(), error=str(exc))
            return finish_trace("Direct", task, [step], None, started, "model_error")
        finally:
            if original_timeout is not None:
                setattr(self.llm, "timeout", original_timeout)
        step = AgentStep(
            step_id="direct-answer",
            step_type="model",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            timestamp=time.time(),
            duration=time.monotonic() - started,
            observation=response.text,
            metadata={"cached": response.cached},
        )
        failure = "budget_exhausted" if response.input_tokens + response.output_tokens > limits.max_tokens else None
        return finish_trace("Direct", task, [step], None if failure else response.text, started, failure)
