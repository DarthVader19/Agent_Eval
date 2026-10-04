# frameworks/langgraph_runner.py
from typing import Any

from agents.langgraph_agent import LangGraphAgent


class LangGraphFrameworkRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.framework_name = "LangGraph"

    def run(self, task: dict[str, Any], toolbox: Any = None, budget: Any = None):
        agent_config = {**self.config.get("agent_config", {}), "type": "langgraph"}
        return LangGraphAgent(agent_config).run(task, toolbox, budget)