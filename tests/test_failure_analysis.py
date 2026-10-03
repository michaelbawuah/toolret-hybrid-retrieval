"""Meaningful boundary, judgment, and tokenizer accounting regressions."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("failure_analysis", Path(__file__).parents[1] / "scripts/analyze_tool_retrieval_failures.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class WordTokenizer:
    truncation_side = "right"

    def num_special_tokens_to_add(self, pair=True):
        return 3 if pair else 2

    def __call__(self, first, second=None, **kwargs):
        if second is None:
            if isinstance(first, str):
                return {"input_ids": list(range(len(first.split())))}
            return {"input_ids": [list(range(len(item.split()))) for item in first]}
        result = {"input_ids": [], "token_type_ids": [], "special_tokens_mask": []}
        for query, document in zip(first, second):
            q, d = len(query.split()), len(document.split())
            while q + d + 3 > kwargs["max_length"]:
                if q > d:
                    q -= 1
                else:
                    d -= 1
            result["input_ids"].append([101] + [1] * q + [102] + [2] * d + [102])
            result["token_type_ids"].append([0] * (q + 2) + [1] * (d + 1))
            result["special_tokens_mask"].append([1] + [0] * q + [1] + [0] * d + [1])
        return result


def fixture():
    ids = ["toolbench_tool_" + str(n) for n in range(12)]
    corpus = {tool: {"id": tool, "source_domain": "toolbench", "text": json.dumps({"name": "name_" + tool, "description": "word " * 4})}
              for tool in ids}
    query = {"id": "toolbench_query_x", "query": "word word word", "domain": "toolbench", "relevant_ids": [ids[0], ids[-1]]}
    row = {"query_id": query["id"], "source_domain": "toolbench", "relevant_ids": query["relevant_ids"], "rerank_k": 11,
           "rankings": {"hybrid": ids, "reranked": ids[1:11] + [ids[0], ids[11]], "bm25": ids, "dense": ids}}
    return [row], {query["id"]: query}, corpus


def test_separates_candidate_miss_from_reranker_boundary_loss():
    cache, queries, corpus = fixture()
    records, groups = analysis.analyze(cache, queries, corpus, WordTokenizer(), 256)
    record = records[0]
    assert record["candidate_label_recall20"] == .5
    assert record["gold_transitions"] == {"left_top10": 1, "outside_top10": 1}
    assert record["outcome"] == "loss"
    assert record["reranked_ndcg10"] == 0
    assert record["gold_tools"][1]["pair_tokenization"] is None
    assert groups["by_source_domain"]["toolbench"]["positive_labels_left_top10"] == 1


def test_pair_token_accounting_excludes_special_tokens_and_detects_query_loss():
    tokenizer = WordTokenizer()
    rows = analysis.exact_pair_tokens(tokenizer, "a b c d e", ["f g h i j k l"], 5, [7], 9)
    assert rows == [{"query_tokens": 5, "document_tokens": 7, "full_pair_tokens": 15,
                     "retained_query_tokens": 3, "retained_document_tokens": 3,
                     "query_truncated": True, "document_truncated": True, "pair_truncated": True,
                     "removed_document_tokens": 4}]
    assert analysis.exact_pair_tokens(tokenizer, "a b", ["c d"], 2, [2], 20)[0]["pair_truncated"] is False


def test_name_hint_is_conservative_text_condition_and_not_relevance():
    assert analysis.possible_name_counterpart("take_image_screenshot", "Web_Capture_Take_Image_Screenshot")
    assert not analysis.possible_name_counterpart("get", "get_weather")
    assert not analysis.possible_name_counterpart("fuel_prices", "carbon_footprint")


@pytest.mark.parametrize("mutation,match", [
    ("qrels", "Qrels mismatch"), ("domain", "Source mismatch"),
    ("membership", "Invalid prefix"), ("cohort", "identical unique"),
])
def test_changed_evidence_fails(mutation, match):
    cache, queries, corpus = fixture()
    if mutation == "qrels":
        queries[cache[0]["query_id"]]["relevant_ids"] = ["toolbench_tool_2"]
    elif mutation == "domain":
        queries[cache[0]["query_id"]]["domain"] = "apigen"
    elif mutation == "membership":
        cache[0]["rankings"]["reranked"][-1] = "unknown"
    else:
        cache.append(cache[0])
    with pytest.raises(ValueError, match=match):
        analysis.analyze(cache, queries, corpus, WordTokenizer(), 256)


def test_ndcg_multi_positive_normalization_and_rank_boundaries():
    assert analysis.ndcg(["a", "x", "b"], {"a", "b"}) == pytest.approx((1 + .5) / (1 + 1 / 1.584962500721156))
    assert analysis.rank_transition(10, 11) == "left_top10"
    assert analysis.rank_transition(11, 10) == "entered_top10"
    with pytest.raises(ValueError, match="candidate membership"):
        analysis.rank_transition(None, 2)
