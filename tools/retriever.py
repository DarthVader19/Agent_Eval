from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalRetriever:
    def __init__(self, corpus: str | Path = "benchmarks/corpus/documents.jsonl"):
        self.corpus = Path(corpus)

    def retrieve(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        if not self.corpus.exists():
            raise FileNotFoundError(f"Retrieval corpus is missing: {self.corpus}")
        terms = set(query.casefold().split())
        ranked: list[tuple[int, dict[str, Any]]] = []
        with self.corpus.open(encoding="utf-8") as file:
            for line in file:
                document = json.loads(line)
                text = f"{document['title']} {document['text']}".casefold()
                score = sum(text.count(term) for term in terms)
                if score:
                    ranked.append((score, document))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [doc for _, doc in ranked[:limit]]
