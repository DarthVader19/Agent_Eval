from __future__ import annotations


class FaultInjector:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.calls: dict[str, int] = {}

    def apply(self, tool: str, output: str) -> str:
        fault = self.config.get(tool)
        if not fault:
            return output
        kind = fault if isinstance(fault, str) else fault.get("type", "")
        count = self.calls.get(tool, 0)
        self.calls[tool] = count + 1
        if kind == "transient" and count == 0:
            raise RuntimeError("Injected transient tool failure.")
        if kind == "permanent":
            raise RuntimeError("Injected permanent tool failure.")
        if kind == "timeout":
            raise TimeoutError("Injected tool timeout.")
        if kind == "empty":
            return ""
        if kind == "malformed":
            return "{malformed tool output"
        if kind == "wrong":
            return "Injected plausible but incorrect result."
        return output
