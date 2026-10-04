from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
import time
from typing import Any

from tools.calculator import calculate
from tools.fault_injection import FaultInjector
from tools.python_sandbox import run_python
from tools.retriever import LocalRetriever
from tools.search import SnapshotSearch


class ToolBox:
    """Shared, cacheable tool interface used by every agent and framework."""

    def __init__(
        self,
        *,
        cache_enabled: bool = True,
        cache_path: str | Path = "results/tool_cache.sqlite3",
        fault_config: dict | None = None,
        allowed_tools: list[str] | None = None,
        corpus_path: str | Path = "benchmarks/corpus/documents.jsonl",
        search_snapshot: str | Path = "benchmarks/corpus/search_snapshots.json",
    ):
        self.cache_enabled = cache_enabled
        self.cache_path = Path(cache_path)
        self.allowed_tools = set(
            allowed_tools if allowed_tools is not None else {"calculator", "python", "search", "retrieve", "read_file"}
        )
        self.faults = FaultInjector(fault_config)
        self.retriever = LocalRetriever(corpus_path)
        self.searcher = SnapshotSearch(search_snapshot)
        self.call_log: list[dict[str, Any]] = []

    def invoke(self, name: str, params: dict[str, Any]) -> str:
        started = time.monotonic()
        if name not in self.allowed_tools:
            message = f"Tool {name!r} is disabled by this experiment."
            self.call_log.append(
                {"tool": name, "input": params, "output": None, "cached": False, "duration": 0.0, "error": message}
            )
            raise ValueError(message)
        cache_version = self._cache_version(name, params)
        key = hashlib.sha256(
            json.dumps([name, params, cache_version], sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if self.cache_enabled and not self.faults.config.get(name):
            cached = self._read_cache(key)
            if cached is not None:
                self.call_log.append({"tool": name, "input": params, "output": cached, "cached": True, "duration": 0.0})
                return cached
        try:
            result = self._execute(name, params)
            result = self.faults.apply(name, result)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            if self.cache_enabled and not self.faults.config.get(name):
                self._write_cache(key, result)
            self.call_log.append({"tool": name, "input": params, "output": result, "cached": False, "duration": time.monotonic() - started})
            return result
        except Exception as exc:
            self.call_log.append({"tool": name, "input": params, "output": None, "cached": False, "duration": time.monotonic() - started, "error": str(exc)})
            raise

    def _execute(self, name: str, params: dict[str, Any]) -> str:
        if name == "calculator":
            return str(calculate(str(params["expression"])))
        if name == "python":
            return run_python(str(params["code"]), float(params.get("timeout", 3)))
        if name == "search":
            return json.dumps(self.searcher.search(str(params["query"])), ensure_ascii=False)
        if name == "retrieve":
            return json.dumps(self.retriever.retrieve(str(params["query"])), ensure_ascii=False)
        if name == "read_file":
            path = Path(str(params["path"])).resolve()
            allowed_root = Path("benchmarks/corpus").resolve()
            if allowed_root not in path.parents and path != allowed_root:
                raise ValueError("File reads are restricted to benchmarks/corpus.")
            return path.read_text(encoding="utf-8")
        raise ValueError(f"Unknown tool: {name}")

    def _cache_version(self, name: str, params: dict[str, Any]) -> str | None:
        if name == "retrieve":
            path = self.retriever.corpus
        elif name == "search":
            path = self.searcher.snapshot
        elif name == "read_file":
            path = Path(str(params["path"]))
        else:
            return None
        if not path.exists():
            return str(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.cache_path, timeout=30)
        connection.execute("CREATE TABLE IF NOT EXISTS tool_cache (cache_key TEXT PRIMARY KEY, output TEXT NOT NULL)")
        return connection

    def _read_cache(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT output FROM tool_cache WHERE cache_key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _write_cache(self, key: str, output: str) -> None:
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO tool_cache VALUES (?, ?)", (key, output))
