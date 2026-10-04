from __future__ import annotations

import json
import time
from typing import Any

from agents.base import AgentStep, AgentTrace, BaseAgent, Budget, estimate_tokens
from tools.toolbox import ToolBox


def execute_step(
    agent: BaseAgent,
    task: dict[str, Any],
    toolbox: ToolBox,
    budget: Budget,
    steps: list[AgentStep],
    context: list[str],
    started: float,
    *,
    planning: str = "",
) -> tuple[str | None, str | None]:
    used_tokens = sum(step.input_tokens + step.output_tokens for step in steps)
    if time.monotonic() - started >= budget.max_seconds or used_tokens >= budget.max_tokens:
        return None, "budget_exhausted"
    allowed_tools = agent.config.get("tools", ["calculator", "python", "search", "retrieve", "read_file"])
    available = "|".join(str(tool) for tool in allowed_tools)
    prompt = (
        f"Task: {task['prompt']}\n"
        f"Plan: {planning or 'Solve the task directly; use a tool if useful.'}\n"
        f"Execution history:\n{chr(10).join(context[-8:]) or '(none)'}\n\n"
        "Choose exactly one JSON action. Use "
        f'{{"type":"tool","tool":"{available}","params":{{...}}}} '
        'or {"type":"final","answer":"..."}. Never invent tool output.'
    )
    remaining_tokens = budget.max_tokens - used_tokens - estimate_tokens(prompt)
    if remaining_tokens <= 0:
        return None, "budget_exhausted"
    max_output_tokens = min(int(agent.config.get("max_output_tokens", 512)), remaining_tokens)
    remaining_seconds = budget.max_seconds - (time.monotonic() - started)
    original_timeout = getattr(agent.llm, "timeout", None)
    if original_timeout is not None:
        setattr(agent.llm, "timeout", min(float(original_timeout), max(0.1, remaining_seconds)))
    model_started = time.monotonic()
    try:
        response = agent._complete(prompt, max_output_tokens=max_output_tokens)
    except Exception as exc:
        steps.append(
            AgentStep(
                step_id=f"model-{len(steps)}",
                step_type="model",
                timestamp=time.time(),
                duration=time.monotonic() - model_started,
                error=str(exc),
            )
        )
        return None, "budget_exhausted" if time.monotonic() - started >= budget.max_seconds else "model_error"
    finally:
        if original_timeout is not None:
            setattr(agent.llm, "timeout", original_timeout)
    action = agent._json_response(response.text)
    steps.append(
        AgentStep(
            step_id=f"model-{len(steps)}",
            step_type="reason",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            timestamp=time.time(),
            duration=time.monotonic() - model_started,
            action=action,
            observation=response.text,
            metadata={"cached": response.cached},
        )
    )
    if used_tokens + response.input_tokens + response.output_tokens > budget.max_tokens:
        return None, "budget_exhausted"
    if action.get("type") == "final":
        return str(action.get("answer", "")), None
    if action.get("type") != "tool":
        context.append(f"Invalid action schema: {json.dumps(action, default=str)}")
        return None, None
    tool_name = str(action.get("tool", ""))
    params = action.get("params", {})
    if not isinstance(params, dict):
        context.append("Tool parameters must be a JSON object.")
        return None, None
    if tool_name == "python":
        remaining_seconds = budget.max_seconds - (time.monotonic() - started)
        if remaining_seconds <= 0:
            return None, "budget_exhausted"
        params["timeout"] = min(float(params.get("timeout", 3)), remaining_seconds)
    tool_started = time.monotonic()
    try:
        output = toolbox.invoke(tool_name, params)
        error = None
    except Exception as exc:
        output = ""
        error = str(exc)
    steps.append(
        AgentStep(
            step_id=f"tool-{len(steps)}",
            step_type="tool",
            timestamp=time.time(),
            duration=time.monotonic() - tool_started,
            action={"tool": tool_name, "params": params},
            observation=output,
            error=error,
            metadata={"tool_input": params, "tool_output": output},
        )
    )
    if error:
        context.append(f"Tool {tool_name} failed: {error}. Retry or choose another action.")
    else:
        context.append(f"Tool {tool_name} returned: {output}")
    return None, None


def finish_trace(
    agent_type: str,
    task: dict[str, Any],
    steps: list[AgentStep],
    answer: str | None,
    started: float,
    budget_reason: str | None = None,
) -> AgentTrace:
    errors = [step.error for step in steps if step.error]
    return AgentTrace(
        agent_type=agent_type,
        task_id=str(task.get("id", "unknown")),
        steps=steps,
        final_answer=answer or "",
        success=answer is not None and budget_reason is None,
        total_tokens=sum(step.input_tokens + step.output_tokens for step in steps),
        total_time=time.monotonic() - started,
        errors=errors,
        failure_reason=budget_reason,
    )
