from math import log2

def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & set(relevant_ids)) / len(relevant_ids)

def reciprocal_rank(ranked_ids, relevant_ids):
    relevant_ids = set(relevant_ids)
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(ranked_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)
    if not relevant_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id in relevant_ids:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0

def evaluate_rankings(rankings, qrels, ks=(1, 5, 10)):
    if not qrels:
        raise ValueError("qrels cannot be empty.")
    qids = list(qrels)
    result = {
        "MRR": sum(
            reciprocal_rank(rankings.get(qid, []), qrels[qid])
            for qid in qids
        ) / len(qids)
    }
    for k in ks:
        result[f"Recall@{k}"] = sum(
            recall_at_k(rankings.get(qid, []), qrels[qid], k)
            for qid in qids
        ) / len(qids)
        result[f"nDCG@{k}"] = sum(
            ndcg_at_k(rankings.get(qid, []), qrels[qid], k)
            for qid in qids
        ) / len(qids)
    return result
