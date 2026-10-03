import pytest

from toolret_research.selective_runtime import route_and_rerank


class Spy:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates):
        self.calls.append((query, candidates))
        return [(doc_id, 1.) for doc_id, _ in reversed(candidates)]


DOCS = {"a": "weather", "b": "calendar", "c": "email"}


def test_agreement_skips_backend_entirely():
    backend = Spy()
    result = route_and_rerank("weather", ["a", "b"], ["a", "c"], ["a", "b", "c"], DOCS, backend)
    assert backend.calls == []
    assert result.ranking == ("a", "b", "c")
    assert result.scored_pairs == 0
    assert not result.reranked


def test_disagreement_scores_only_prefix_and_preserves_tail():
    backend = Spy()
    result = route_and_rerank("weather", ["a", "b"], ["b", "a"], ["a", "b", "c"], DOCS, backend, depth=2)
    assert backend.calls == [("weather", [("a", "weather"), ("b", "calendar")])]
    assert result.ranking == ("b", "a", "c")
    assert result.scored_pairs == 2
    assert result.reranked


def test_backend_cannot_invent_or_drop_candidates():
    class Bad:
        def rerank(self, query, candidates):
            return [("c", 1.)]
    with pytest.raises(ValueError, match="permutation"):
        route_and_rerank("weather", ["a"], ["b"], ["a", "b", "c"], DOCS, Bad(), depth=2)


@pytest.mark.parametrize("depth", [0, -1, True, 1.5])
def test_invalid_depth_fails_before_backend(depth):
    with pytest.raises(ValueError):
        route_and_rerank("weather", ["a"], ["b"], ["a", "b"], DOCS, Spy(), depth=depth)
