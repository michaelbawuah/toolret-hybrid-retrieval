from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log

from .text import tokenize

@dataclass
class BM25Index:
    doc_ids: list[str]
    doc_lengths: list[int]
    term_freqs: list[Counter]
    doc_freqs: dict[str, int]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, documents, tokenizer=tokenize, k1=1.5, b=0.75):
        if not documents:
            raise ValueError("Cannot build BM25 index over an empty corpus.")

        doc_ids = list(documents.keys())
        term_freqs = []
        doc_lengths = []
        doc_freqs = defaultdict(int)

        for doc_id in doc_ids:
            tokens = tokenizer(documents[doc_id])
            tf = Counter(tokens)
            term_freqs.append(tf)
            doc_lengths.append(len(tokens))
            for term in tf:
                doc_freqs[term] += 1

        return cls(
            doc_ids=doc_ids,
            doc_lengths=doc_lengths,
            term_freqs=term_freqs,
            doc_freqs=dict(doc_freqs),
            avgdl=sum(doc_lengths) / len(doc_lengths),
            k1=k1,
            b=b,
        )

    def idf(self, term):
        n = len(self.doc_ids)
        df = self.doc_freqs.get(term, 0)
        return log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_tokens(self, query_tokens):
        scores = [0.0] * len(self.doc_ids)
        for i, (tf, dl) in enumerate(zip(self.term_freqs, self.doc_lengths)):
            norm = self.k1 * (
                1.0 - self.b + self.b * dl / max(self.avgdl, 1e-12)
            )
            score = 0.0
            for term in query_tokens:
                freq = tf.get(term, 0)
                if freq:
                    score += self.idf(term) * (
                        freq * (self.k1 + 1.0)
                    ) / (freq + norm)
            scores[i] = score
        return scores

    def search(self, query, k=10, tokenizer=tokenize):
        if k <= 0:
            return []
        scores = self.score_tokens(tokenizer(query))
        ranked = sorted(
            zip(self.doc_ids, scores),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[: min(k, len(ranked))]
