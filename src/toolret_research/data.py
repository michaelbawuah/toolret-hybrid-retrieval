from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class QueryExample:
    id: str
    text: str
    relevant_ids: frozenset[str]

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from e
    return rows

def load_corpus(path):
    corpus = {}
    for row in read_jsonl(path):
        if "id" not in row:
            raise ValueError("Each corpus row must contain 'id'.")
        doc_id = str(row["id"])
        if doc_id in corpus:
            raise ValueError(f"Duplicate corpus id: {doc_id}")
        corpus[doc_id] = row
    return corpus

def load_queries(path):
    out = []
    for row in read_jsonl(path):
        if "id" not in row or "query" not in row:
            raise ValueError("Each query row must contain 'id' and 'query'.")
        relevant = set()
        for label in row.get("labels", []):
            if "id" in label and float(label.get("relevance", 0)) > 0:
                relevant.add(str(label["id"]))
        out.append(QueryExample(
            id=str(row["id"]),
            text=str(row["query"]),
            relevant_ids=frozenset(relevant),
        ))
    return out
