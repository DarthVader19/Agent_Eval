from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, Protocol


@dataclass
class AgentStep:
    step_id: str
    step_type: str
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    action: dict[str, Any] | None = None
    observation: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    agent_type: str
    task_id: str
    steps: list[AgentStep]
    final_answer: Any
    success: bool
    total_tokens: int
    total_time: float
    errors: list[str]
    reflection_used: bool = False
    reflection_steps: list[AgentStep] = field(default_factory=list)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Budget:
    max_steps: int = 10
    max_tokens: int = 4000
    max_seconds: float = 90.0


@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cached: bool = False


class ChatModel(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_output_tokens: int = 512,
    ) -> ModelResponse: ...


class BaseAgent(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.agent_type = self.__class__.__name__
        self.llm: ChatModel = config.get("llm_client") or _create_model(config)

    @abstractmethod
    def run(
        self,
        task: dict[str, Any],
        toolbox: Any = None,
        budget: Budget | dict[str, Any] | None = None,
    ) -> AgentTrace:
        raise NotImplementedError

    def _budget(self, task: dict[str, Any], budget: Budget | dict[str, Any] | None) -> Budget:
        if isinstance(budget, Budget):
            return budget
        values = budget or {}
        task_steps = task.get("max_steps")
        return Budget(
            max_steps=int(values.get("max_steps", task_steps or self.config.get("max_steps", 10))),
            max_tokens=int(values.get("max_tokens", self.config.get("max_tokens", 4000))),
            max_seconds=float(values.get("max_seconds", self.config.get("max_seconds", 90))),
        )

    def _complete(self, prompt: str, *, max_output_tokens: int | None = None) -> ModelResponse:
        return self.llm.complete(
            prompt,
            system=str(self.config.get("system_prompt", "You are a careful assistant. Use tools when useful. Return concise answers.")),
            temperature=float(self.config.get("temperature", 0.0)),
            max_output_tokens=max_output_tokens or int(self.config.get("max_output_tokens", 512)),
        )

    @staticmethod
    def _json_response(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    value = json.loads(text[start:end])
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    pass
            return {"type": "final", "answer": text.strip()}

    @staticmethod
    def _failed_trace(agent: str, task: dict[str, Any], steps: list[AgentStep], started: float, reason: str) -> AgentTrace:
        return AgentTrace(
            agent_type=agent,
            task_id=str(task.get("id", "unknown")),
            steps=steps,
            final_answer="",
            success=False,
            total_tokens=sum(s.input_tokens + s.output_tokens for s in steps),
            total_time=time.monotonic() - started,
            errors=[s.error for s in steps if s.error],
            failure_reason=reason,
        )


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _create_model(config: dict[str, Any]) -> ChatModel:
    from harness.model import OpenAICompatibleModel

    return OpenAICompatibleModel(
        model=str(config.get("model", "gpt-4o-mini")),
        cache_path=config.get("cache_path", "results/cache.sqlite3"),
        cache_enabled=bool(config.get("cache_enabled", True)),
        timeout=float(config.get("request_timeout", 60)),
    )
