from __future__ import annotations

import asyncio
from importlib import import_module
import os
import time
from typing import Any

from agents.base import AgentStep, BaseAgent, Budget, estimate_tokens
from tools.toolbox import ToolBox


class AutoGenAdapter(BaseAgent):
    """AutoGen single-worker mapping over the benchmark's shared ToolBox."""

    def run(self, task: dict[str, Any], toolbox: ToolBox | None = None, budget: Budget | dict[str, Any] | None = None):
        limits = self._budget(task, budget)
        tools = toolbox or ToolBox()
        started = time.monotonic()
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import RoundRobinGroupChat
            OpenAIChatCompletionClient = import_module("autogen_ext.models.openai").OpenAIChatCompletionClient
        except ImportError as exc:
            raise RuntimeError(
                "AutoGen OpenAI adapter dependencies are missing; install requirements-frameworks.txt."
            ) from exc
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before using the AutoGen framework adapter.")
        model = str(self.config.get("model", "gpt-4o-mini"))
        model_options: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": float(self.config.get("temperature", 0.0)),
            "max_tokens": int(self.config.get("max_output_tokens", 512)),
            "timeout": limits.max_seconds,
        }
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            model_options["base_url"] = base_url
        client = OpenAIChatCompletionClient(**model_options)

        def calculator(expression: str) -> str:
            return tools.invoke("calculator", {"expression": expression})

        def python(code: str) -> str:
            return tools.invoke("python", {"code": code})

        def search(query: str) -> str:
            return tools.invoke("search", {"query": query})

        def retrieve(query: str) -> str:
            return tools.invoke("retrieve", {"query": query})

        def read_file(path: str) -> str:
            return tools.invoke("read_file", {"path": path})

        worker = AssistantAgent(
            name="evaluation_worker",
            model_client=client,
            tools=[calculator, python, search, retrieve, read_file],
            system_message="Solve the task with the provided shared tools. Return a concise final answer.",
            max_tool_iterations=limits.max_steps,
        )
        team = RoundRobinGroupChat(
            [worker],
            termination_condition=MaxMessageTermination(max_messages=max(2, limits.max_steps * 2)),
            max_turns=max(1, limits.max_steps),
        )

        async def execute():
            try:
                return await team.run(task=task["prompt"])
            finally:
                await client.close()

        result = asyncio.run(execute())
        answer = ""
        for message in reversed(result.messages):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                answer = content
                break
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
                step_id="autogen-final",
                step_type="final",
                input_tokens=estimate_tokens(task["prompt"]),
                output_tokens=estimate_tokens(answer),
                timestamp=time.time(),
                duration=time.monotonic() - started,
                observation=answer,
            )
        )
        from agents.base import AgentTrace

        return AgentTrace(
            agent_type="AutoGen",
            task_id=str(task.get("id", "unknown")),
            steps=steps,
            final_answer=answer,
            success=bool(answer),
            total_tokens=sum(step.input_tokens + step.output_tokens for step in steps),
            total_time=time.monotonic() - started,
            errors=[step.error for step in steps if step.error],
        )
