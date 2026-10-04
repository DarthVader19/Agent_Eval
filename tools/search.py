from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SnapshotSearch:
    def __init__(self, snapshot: str | Path = "benchmarks/corpus/search_snapshots.json"):
        self.snapshot = Path(snapshot)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self.snapshot.exists():
            raise FileNotFoundError(f"Search snapshot is missing: {self.snapshot}")
        documents = json.loads(self.snapshot.read_text(encoding="utf-8"))
        terms = set(query.casefold().split())
        ranked = []
        for document in documents:
            text = f"{document['title']} {document['text']}".casefold()
            rank = sum(text.count(term) for term in terms)
            if rank:
                ranked.append((rank, document))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [document for _, document in ranked[:limit]]
