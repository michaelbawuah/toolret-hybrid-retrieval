import math

import numpy as np
import pytest

from toolret_research.utility_router import (
    FEATURE_NAMES, RidgeUtilityModel, calibratebudgetthreshold, features,
    fitmodel, predict, shouldroute,
)


def rankings():
    return {"bm25": [f"tool-{i}" for i in range(100)],
            "dense": [f"tool-{i}" for i in reversed(range(100))]}


def test_features_do_not_use_identity_semantics_or_expensive_rankings():
    original = rankings()
    original["reranked"] = ["secret-gold-id"]
    original["relevant_ids"] = ["secret-label"]
    renamed = {method: ["renamed-" + item for item in values]
               for method, values in original.items()}
    renamed["reranked"] = ["completely-different-expensive-output"]
    renamed["relevant_ids"] = []
    before = features("find some tools", original)
    after = features("other words here", renamed)
    np.testing.assert_array_equal(before, after)


def test_identical_rankings_have_exact_overlap_and_known_rrf_margin():
    ids = [str(i) for i in range(100)]
    values = features("two words", {"bm25": ids, "dense": ids})
    np.testing.assert_allclose(values[:5], [0, 1, 1, 1, 1], rtol=0, atol=0)
    assert values[5] == math.log1p(2)
    assert values[6] == pytest.approx((2 / 61 - 2 / 62) / (2 / 61))


def test_retriever_swap_does_not_change_features():
    original = rankings()
    np.testing.assert_array_equal(features("find tools", original), features("find tools", {
        "bm25": original["dense"], "dense": original["bm25"]}))


@pytest.mark.parametrize("bad", [[], ["a", "a"], [""], [1], "abc", None])
def test_invalid_first_stage_rankings_fail(bad):
    with pytest.raises(ValueError):
        features("query", {"bm25": bad, "dense": ["b"]})


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_invalid_query_fails(bad):
    with pytest.raises(ValueError):
        features(bad, rankings())


def test_fixed_alpha_ridge_matches_closed_form_and_training_only_scales():
    x = np.zeros((4, len(FEATURE_NAMES)))
    x[:, 0] = [-1, -1, 1, 1]
    x[:, 1:] = 3.
    y = np.asarray([-1., -1., 1., 1.])
    model = fitmodel(x, y, alpha=10)
    assert model.alpha == 10
    assert model.training_rows == 4
    assert model.means == (0., 3., 3., 3., 3., 3., 3.)
    assert model.scales == (1., 1., 1., 1., 1., 1., 1.)
    assert model.intercept == 0
    assert model.coefficients[0] == pytest.approx(4 / 14)
    np.testing.assert_allclose(predict(model, x), y * (4 / 14))
    # Prediction at a shifted distribution does not refit means or variance.
    shifted = x.copy()
    shifted[:, 0] += 10
    np.testing.assert_allclose(predict(model, shifted), (y + 10) * (4 / 14))
    assert model.means[0] == 0


def test_constant_features_and_target_keep_unpenalized_intercept():
    x = np.ones((5, len(FEATURE_NAMES)))
    model = fitmodel(x, np.full(5, .25))
    assert model.coefficients == (0.,) * len(FEATURE_NAMES)
    assert model.intercept == .25
    assert predict(model, x[0]) == .25
    np.testing.assert_array_equal(predict(model, x), np.full(5, .25))


def test_model_serialization_preserves_predictions_and_feature_order():
    x = np.stack([features("find tools", rankings()), features("find many tools", {
        "bm25": ["a", "b"], "dense": ["a", "b"]})])
    model = fitmodel(x, [.1, -.1])
    loaded = RidgeUtilityModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(predict(model, x), predict(loaded, x))
    altered = model.to_dict()
    altered["feature_names"] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError, match="feature order"):
        RidgeUtilityModel.from_dict(altered)


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf"), True])
def test_nonpositive_or_nonfinite_regularization_fails(alpha):
    with pytest.raises(ValueError):
        fitmodel(np.ones((2, len(FEATURE_NAMES))), [0, 1], alpha)


@pytest.mark.parametrize("target", [[0], [0, float("nan")], [-1.1, 0], [0, 1.1]])
def test_invalid_or_misaligned_training_targets_fail(target):
    with pytest.raises(ValueError):
        fitmodel(np.ones((2, len(FEATURE_NAMES))), target)


def test_invalid_feature_matrices_fail_before_fit():
    with pytest.raises(ValueError):
        fitmodel(np.ones((2, 6)), [0, 1])
    with pytest.raises(ValueError):
        fitmodel(np.full((2, 7), np.nan), [0, 1])
    with pytest.raises(ValueError):
        fitmodel(np.ones((1, 7)), [0])


def test_threshold_is_linear_prediction_quantile_with_strict_ties():
    scores = np.asarray([0., 1., 2., 3.])
    threshold = calibratebudgetthreshold(scores, .75)
    assert threshold == .75
    assert [shouldroute(score, threshold) for score in scores] == [False, True, True, True]
    tied = np.asarray([1., 1., 1., 1.])
    tied_threshold = calibratebudgetthreshold(tied, .75)
    assert tied_threshold == 1.
    assert not any(shouldroute(score, tied_threshold) for score in tied)


def test_future_decision_is_pointwise_and_does_not_force_test_budget():
    threshold = calibratebudgetthreshold([0., 1., 2., 3.], .5)
    assert threshold == 1.5
    assert all(shouldroute(score, threshold) for score in [2., 3., 4., 5.])
    assert not any(shouldroute(score, threshold) for score in [-4., -3., -2., -1.])


@pytest.mark.parametrize("scores,budget", [([], .5), ([np.nan], .5), ([[1, 2]], .5),
                                          ([1, 2], 0), ([1, 2], 1), ([1, 2], True)])
def test_invalid_calibration_input_fails(scores, budget):
    with pytest.raises(ValueError):
        calibratebudgetthreshold(scores, budget)


@pytest.mark.parametrize("score,threshold", [(math.nan, 0), (0, math.inf), (math.inf, 0)])
def test_invalid_runtime_score_fails(score, threshold):
    with pytest.raises(ValueError):
        shouldroute(score, threshold)
