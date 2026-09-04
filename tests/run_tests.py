import json
import tempfile
from pathlib import Path

from toolret_research.bm25 import BM25Index
from toolret_research.data import load_corpus, load_queries
from toolret_research.metrics import recall_at_k, reciprocal_rank, ndcg_at_k, evaluate_rankings

def approx(a, b, eps=1e-9):
    return abs(a-b) < eps

def test_bm25():
    docs = {
        "weather": "weather forecast temperature city",
        "stocks": "stock market historical price ticker",
        "math": "calculate multiplication arithmetic numbers",
    }
    idx = BM25Index.build(docs)
    hits = idx.search("forecast city weather", k=2)
    assert hits[0][0] == "weather"
    assert hits[0][1] > hits[1][1]

def test_metrics():
    assert approx(recall_at_k(["a","b","c"], {"b","d"}, 2), 0.5)
    assert approx(reciprocal_rank(["x","b","a"], {"a","b"}), 0.5)
    assert approx(ndcg_at_k(["a","b","x"], {"a","b"}, 3), 1.0)
    out = evaluate_rankings(
        {"q1":["a","b"], "q2":["x","c"]},
        {"q1":{"a"}, "q2":{"c"}},
        ks=[1,2],
    )
    assert approx(out["Recall@1"], 0.5)
    assert approx(out["MRR"], 0.75)

def test_data():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cp = td/"corpus.jsonl"
        qp = td/"queries.jsonl"
        cp.write_text(json.dumps({"id":"t1","name":"weather"})+"\n", encoding="utf-8")
        qp.write_text(json.dumps({
            "id":"q1",
            "query":"weather tomorrow",
            "labels":[{"id":"t1","relevance":1,"doc":{"name":"weather"}}]
        })+"\n", encoding="utf-8")
        c = load_corpus(cp)
        q = load_queries(qp)
        assert "t1" in c
        assert q[0].relevant_ids == frozenset({"t1"})

if __name__ == "__main__":
    tests = [test_bm25, test_metrics, test_data]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
