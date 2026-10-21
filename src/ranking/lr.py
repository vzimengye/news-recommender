"""Logistic regression ranker on standardized candidate features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticRegressionRanker:
    """Standardize features, then fit sklearn's LogisticRegression.

    Exposes ``fit``/``predict_proba`` like the project's other models and
    retrievers, so callers (scripts/evaluate.py) can treat every scorer the
    same way.
    """

    def __init__(self, **lr_kwargs: object) -> None:
        lr_kwargs.setdefault("max_iter", 1000)
        self._scaler = StandardScaler()
        self._model = LogisticRegression(**lr_kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionRanker":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted probability of a click for each row."""
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]


def downsample_negatives(
    X: pd.DataFrame, y: pd.Series, *, negative_ratio: float, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep every positive row and a random subset of negatives.

    ``negative_ratio`` is negatives kept per positive (e.g. 4.0 keeps about
    four negatives for every positive). Returns the same rows unchanged,
    just fewer of them -- ranking metrics are invariant to this kind of
    class-balance shift since it only rescales scores monotonically, not
    their relative order, so no intercept correction is needed for ranking
    (it would matter if calibrated probabilities were needed instead).
    """
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")

    rng = np.random.default_rng(seed)
    pos_mask = (y == 1).to_numpy()
    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(~pos_mask)

    n_keep_neg = min(len(neg_idx), int(round(len(pos_idx) * negative_ratio)))
    sampled_neg_idx = rng.choice(neg_idx, size=n_keep_neg, replace=False)
    keep_idx = np.sort(np.concatenate([pos_idx, sampled_neg_idx]))
    return X.iloc[keep_idx], y.iloc[keep_idx]
