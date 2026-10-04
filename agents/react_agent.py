from __future__ import annotations

import time
from typing import Any

from agents.base import BaseAgent, Budget
from agents.runtime import execute_step, finish_trace
from tools.toolbox import ToolBox


class ReActAgent(BaseAgent):
    """Instrumented reason-act-observe loop using the shared ToolBox."""

    def run(
        self,
        task: dict[str, Any],
        toolbox: ToolBox | None = None,
        budget: Budget | dict[str, Any] | None = None,
    ):
        limits = self._budget(task, budget)
        tools = toolbox or ToolBox(cache_enabled=bool(self.config.get("cache_enabled", True)))
        started = time.monotonic()
        steps = []
        context: list[str] = []
        answer = None
        failure = None
        for _ in range(limits.max_steps):
            answer, failure = execute_step(self, task, tools, limits, steps, context, started)
            if answer is not None or failure is not None:
                break
        if answer is None and failure is None and len(steps) >= limits.max_steps:
            failure = "budget_exhausted"
        return finish_trace("ReAct", task, steps, answer, started, failure)
