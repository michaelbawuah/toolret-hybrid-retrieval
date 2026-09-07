from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HardNegativeExample:
    """
    One training example for retrieval fine-tuning.

    Each example contains:
    - the original query
    - one known relevant/positive tool
    - several difficult but incorrect tools
    """

    query_id: str
    query: str
    positive_id: str
    negative_ids: tuple[str, ...]


def mine_hard_negatives(
    query_id: str,
    query: str,
    relevant_ids: Iterable[str],
    ranked_results: Iterable,
    num_negatives: int = 5,
) -> list[HardNegativeExample]:
    """
    Mine hard negatives from a retriever's ranked results.

    A hard negative is a highly ranked document that is NOT
    one of the known relevant documents.

    One HardNegativeExample is produced for each relevant tool.
    """

    if num_negatives <= 0:
        raise ValueError("num_negatives must be greater than 0.")

    relevant = {str(doc_id) for doc_id in relevant_ids}

    if not relevant:
        return []

    negatives = []

    for result in ranked_results:

        # Our retrievers normally return tuples such as:
        # ("tool_id", score)
        if isinstance(result, (tuple, list)):
            doc_id = str(result[0])
        else:
            doc_id = str(result)

        # Never use a relevant document as a negative.
        if doc_id in relevant:
            continue

        # Avoid duplicate negatives.
        if doc_id in negatives:
            continue

        negatives.append(doc_id)

        if len(negatives) >= num_negatives:
            break

    # If the retriever did not return enough incorrect documents,
    # do not create an incomplete training example.
    if len(negatives) < num_negatives:
        return []

    examples = []

    for positive_id in sorted(relevant):
        examples.append(
            HardNegativeExample(
                query_id=str(query_id),
                query=str(query),
                positive_id=positive_id,
                negative_ids=tuple(negatives),
            )
        )

    return examples


def mine_dataset_hard_negatives(
    queries,
    rankings,
    num_negatives: int = 5,
) -> list[HardNegativeExample]:
    """
    Mine hard-negative examples for an entire query dataset.

    Parameters
    ----------
    queries:
        QueryExample objects from data.py.

    rankings:
        Dictionary:
            query_id -> ranked retrieval results

    num_negatives:
        Number of hard negatives to keep per training example.
    """

    examples = []

    for query in queries:
        ranked_results = rankings.get(query.id, [])

        query_examples = mine_hard_negatives(
            query_id=query.id,
            query=query.text,
            relevant_ids=query.relevant_ids,
            ranked_results=ranked_results,
            num_negatives=num_negatives,
        )

        examples.extend(query_examples)

    return examples