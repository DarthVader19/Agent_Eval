from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.base import AgentStep, BaseAgent, Budget
from agents.runtime import execute_step, finish_trace
from tools.toolbox import ToolBox


class AgentState(TypedDict):
    task: dict[str, Any]
    steps: list[AgentStep]
    context: list[str]
    answer: str | None
    failure: str | None
    iteration: int
    started: float


class LangGraphAgent(BaseAgent):
    """LangGraph StateGraph loop with conditional termination and in-memory checkpointing."""

    def run(
        self,
        task: dict[str, Any],
        toolbox: ToolBox | None = None,
        budget: Budget | dict[str, Any] | None = None,
    ):
        limits = self._budget(task, budget)
        tools = toolbox or ToolBox(cache_enabled=bool(self.config.get("cache_enabled", True)))
        started = time.monotonic()

        def act(state: AgentState) -> dict[str, Any]:
            steps = list(state["steps"])
            context = list(state["context"])
            answer, failure = execute_step(
                self, state["task"], tools, limits, steps, context, state["started"]
            )
            return {
                "steps": steps,
                "context": context,
                "answer": answer,
                "failure": failure,
                "iteration": state["iteration"] + 1,
            }

        def route(state: AgentState) -> str:
            if state["answer"] is not None or state["failure"] is not None:
                return END
            if state["iteration"] >= limits.max_steps:
                return END
            return "act"

        graph = StateGraph(AgentState)
        graph.add_node("act", act)
        graph.set_entry_point("act")
        graph.add_conditional_edges("act", route, {"act": "act", END: END})
        app = graph.compile(checkpointer=MemorySaver())
        final_state = app.invoke(
            {
                "task": task,
                "steps": [],
                "context": [],
                "answer": None,
                "failure": None,
                "iteration": 0,
                "started": started,
            },
            config={"configurable": {"thread_id": str(task.get("id", "task"))}},
        )
        failure = final_state["failure"]
        if final_state["answer"] is None and failure is None:
            failure = "budget_exhausted"
        steps = final_state["steps"]
        return finish_trace("LangGraph", task, steps, final_state["answer"], started, failure)
