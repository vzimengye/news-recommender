"""Tests for src.ranking.lr."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ranking.lr import LogisticRegressionRanker, downsample_negatives


def test_logistic_regression_ranker_ranks_separable_data_correctly() -> None:
    X = pd.DataFrame({"x": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    ranker = LogisticRegressionRanker().fit(X, y)
    scores = ranker.predict_proba(X)

    assert (scores[3:] > 0.5).all()
    assert (scores[:3] < 0.5).all()
    assert scores[3] > scores[0]


def test_downsample_negatives_keeps_all_positives_and_requested_ratio() -> None:
    X = pd.DataFrame({"x": range(110)})
    y = pd.Series([1] * 10 + [0] * 100)

    X_small, y_small = downsample_negatives(X, y, negative_ratio=2.0, seed=0)

    assert (y_small == 1).sum() == 10
    assert (y_small == 0).sum() == 20
    assert set(X.loc[(y == 1).to_numpy(), "x"]) <= set(X_small["x"])


def test_downsample_negatives_caps_at_available_negatives() -> None:
    X = pd.DataFrame({"x": range(15)})
    y = pd.Series([1] * 10 + [0] * 5)

    _, y_small = downsample_negatives(X, y, negative_ratio=10.0, seed=0)

    assert (y_small == 0).sum() == 5
    assert (y_small == 1).sum() == 10


def test_downsample_negatives_rejects_non_positive_ratio() -> None:
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([1, 0])

    with pytest.raises(ValueError, match="negative_ratio must be positive"):
        downsample_negatives(X, y, negative_ratio=0.0)
