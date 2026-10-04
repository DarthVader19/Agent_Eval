from __future__ import annotations

import time
from typing import Any

from agents.base import AgentStep, BaseAgent, Budget, estimate_tokens
from tools.toolbox import ToolBox


class CrewAIAdapter(BaseAgent):
    """CrewAI single-worker mapping with tools delegated to the shared ToolBox."""

    def run(self, task: dict[str, Any], toolbox: ToolBox | None = None, budget: Budget | dict[str, Any] | None = None):
        started = time.monotonic()
        limits = self._budget(task, budget)
        tools = toolbox or ToolBox()
        try:
            from crewai import Agent, Crew, LLM, Process, Task
            from crewai.tools import tool
        except Exception as exc:
            raise RuntimeError(
                "CrewAI adapter could not be loaded. Install requirements-frameworks.txt and use Python 3.11-3.13."
            ) from exc

        @tool("calculator")
        def calculator(expression: str) -> str:
            """Evaluate a basic arithmetic expression."""
            return tools.invoke("calculator", {"expression": expression})

        @tool("python")
        def python(code: str) -> str:
            """Run a restricted Python code snippet."""
            return tools.invoke("python", {"code": code})

        @tool("search")
        def search(query: str) -> str:
            """Search fixed cached web-search snapshots."""
            return tools.invoke("search", {"query": query})

        @tool("retrieve")
        def retrieve(query: str) -> str:
            """Retrieve relevant facts from the fixed local corpus."""
            return tools.invoke("retrieve", {"query": query})

        @tool("read_file")
        def read_file(path: str) -> str:
            """Read a file under the fixed benchmark corpus."""
            return tools.invoke("read_file", {"path": path})

        model = str(self.config.get("model", "gpt-4o-mini"))
        llm_options: dict[str, Any] = {
            "model": f"openai/{model}",
            "temperature": float(self.config.get("temperature", 0.0)),
            "max_tokens": int(self.config.get("max_output_tokens", 512)),
            "timeout": limits.max_seconds,
        }
        import os

        if os.getenv("OPENAI_BASE_URL"):
            llm_options["base_url"] = os.environ["OPENAI_BASE_URL"]
        if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
            llm_options["api_key"] = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        worker = Agent(
            role="Evaluation worker",
            goal="Answer the assigned task accurately with the available shared tools.",
            backstory="A single worker using only the benchmark's shared tools.",
            llm=LLM(**llm_options),
            tools=[calculator, python, search, retrieve, read_file],
            allow_delegation=False,
            max_iter=limits.max_steps,
            max_execution_time=limits.max_seconds,
            verbose=False,
        )
        crew_task = Task(
            description=f"{task['prompt']}\nUse the tools if necessary. Return only the final answer.",
            expected_output="A concise final answer.",
            agent=worker,
        )
        output = Crew(agents=[worker], tasks=[crew_task], process=Process.sequential, verbose=False).kickoff()
        answer = str(getattr(output, "raw", output))
        steps = [
            AgentStep(
                step_id=f"tool-{index}",
                step_type="tool",
                timestamp=started,
                duration=float(call.get("duration", 0)),
                action={"tool": call["tool"], "params": call["input"]},
                observation=call.get("output"),
                error=call.get("error"),
                metadata={"tool_input": call["input"], "tool_output": call.get("output")},
            )
            for index, call in enumerate(tools.call_log)
        ]
        steps.append(
            AgentStep(
                step_id="crew-final",
                step_type="final",
                input_tokens=estimate_tokens(task["prompt"]),
                output_tokens=estimate_tokens(answer),
                timestamp=time.time(),
                duration=time.monotonic() - started,
                observation=answer,
            )
        )
        return _trace(task, steps, answer, started, "CrewAI")


def _trace(task: dict[str, Any], steps: list[AgentStep], answer: str, started: float, name: str):
    from agents.base import AgentTrace

    return AgentTrace(
        agent_type=name,
        task_id=str(task.get("id", "unknown")),
        steps=steps,
        final_answer=answer,
        success=bool(answer),
        total_tokens=sum(step.input_tokens + step.output_tokens for step in steps),
        total_time=time.monotonic() - started,
        errors=[step.error for step in steps if step.error],
    )
