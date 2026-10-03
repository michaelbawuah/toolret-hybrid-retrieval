import numpy as np
import pytest
from toolret_research.utility_router import fitmodel
from toolret_research.utility_runtime import route_utility_and_rerank


class Backend:
    def __init__(self, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def rerank(self, query, candidates):
        self.calls += 1
        return [(doc, float("nan") if self.invalid else float(i)) for i, (doc, _) in enumerate(candidates)]


def run(threshold, backend, **kwargs):
    model = fitmodel(np.zeros((2, 7)), np.zeros(2))
    return route_utility_and_rerank("find tools", ["a", "b", "c"], ["b", "a", "c"], ["a", "b", "c"], {"a": "A", "b": "B", "c": "C"}, model, threshold, backend, depth=kwargs.get("depth", 2))


def test_skipped_backend_is_never_invoked_even_on_disagreement():
    backend = Backend()
    result = run(0, backend)
    assert backend.calls == 0 and result.ranking == ("a", "b", "c") and result.scored_pairs == 0


def test_routed_scores_only_prefix_and_preserves_tail():
    backend = Backend()
    result = run(-1, backend)
    assert backend.calls == 1 and result.ranking == ("b", "a", "c") and result.scored_pairs == 2


def test_rejects_nonfinite_backend_scores():
    with pytest.raises(ValueError, match="finite"):
        run(-1, Backend(invalid=True))


@pytest.mark.parametrize("depth", [0, -1, True, 1.5])
def test_rejects_invalid_depth(depth):
    with pytest.raises(ValueError, match="positive integer"):
        run(-1, Backend(), depth=depth)


def test_rejects_malformed_backend_permutation():
    class DuplicateBackend:
        def rerank(self, query, candidates):
            return [(candidates[0][0], 1), (candidates[0][0], 2)]
    with pytest.raises(ValueError, match="exactly once"):
        run(-1, DuplicateBackend())
