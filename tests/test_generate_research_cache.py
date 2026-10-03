import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("generate_cache", Path(__file__).parents[1] / "scripts/generate_research_cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prefix_sort_preserves_tail_and_breaks_ties_by_id():
    assert module.reranked_prefix(["b", "a", "c", "d"], [1., 1., 0.], 3) == ["a", "b", "c", "d"]


@pytest.mark.parametrize("scores,depth", [([1.], 2), ([float('nan')], 1), ([float('inf')], 1)])
def test_incomplete_or_nonfinite_reranker_outputs_fail(scores, depth):
    with pytest.raises(ValueError):
        module.reranked_prefix(["a", "b"], scores, depth)
