"""Ranking and retrieval metrics for news recommendation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from math import log2
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC for one impression's candidates."""
    return float(roc_auc_score(labels, scores))


def _dcg_at_k(labels: Sequence[int], k: int) -> float:
    """Compute DCG for an already ranked sequence of binary relevance labels."""
    return sum(
        (2**int(relevance) - 1) / log2(rank + 1)
        for rank, relevance in enumerate(labels[:k], start=1)
    )


def _ndcg_at_k(ranked_labels: Sequence[int], k: int) -> float:
    dcg = _dcg_at_k(ranked_labels, k)
    ideal_labels = sorted(ranked_labels, reverse=True)
    ideal_dcg = _dcg_at_k(ideal_labels, k)
    return dcg / ideal_dcg if ideal_dcg else 0.0


def ranking_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    """Return impression-averaged AUC, MRR, and nDCG ranking metrics.

    Single-class impressions are excluded from every mean because their AUC is
    undefined. ``n_impressions`` counts all input impressions; ``n_skipped``
    counts the excluded single-class impressions. If there are no eligible
    impressions, metric values are NaN.
    """
    required = {"impression_id", "label", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if df[["impression_id", "label", "score"]].isna().any().any():
        raise ValueError("impression_id, label, and score must not contain missing values")

    labels = df["label"]
    if not labels.isin([0, 1]).all():
        raise ValueError("labels must contain only 0 or 1")
    scores = pd.to_numeric(df["score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("scores must be finite numeric values")

    auc_values: list[float] = []
    mrr_values: list[float] = []
    ndcg5_values: list[float] = []
    ndcg10_values: list[float] = []
    n_skipped = 0

    # sort=False preserves first-seen impression order; stable score ordering
    # makes tied scores deterministic by retaining their original row order.
    for _, group in df.assign(_metric_score=scores).groupby(
        "impression_id", sort=False, dropna=False
    ):
        group_labels = group["label"].to_numpy(dtype=int)
        if np.unique(group_labels).size < 2:
            n_skipped += 1
            continue

        group_scores = group["_metric_score"].to_numpy(dtype=float)
        order = np.argsort(-group_scores, kind="stable")
        ranked_labels = group_labels[order]

        auc_values.append(_roc_auc(group_labels, group_scores))
        first_click_rank = int(np.flatnonzero(ranked_labels == 1)[0]) + 1
        mrr_values.append(1.0 / first_click_rank)
        ndcg5_values.append(_ndcg_at_k(ranked_labels, 5))
        ndcg10_values.append(_ndcg_at_k(ranked_labels, 10))

    n_impressions = int(df["impression_id"].nunique(dropna=False))

    def mean_or_nan(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "auc": mean_or_nan(auc_values),
        "mrr": mean_or_nan(mrr_values),
        "ndcg@5": mean_or_nan(ndcg5_values),
        "ndcg@10": mean_or_nan(ndcg10_values),
        "n_impressions": n_impressions,
        "n_skipped": n_skipped,
    }


def recall_at_k(
    queries: Iterable[tuple[Any, Any, str]],
    retrieve_fn: Callable[[Any, Any, int], Sequence[str]],
    k: int,
) -> float:
    """Return the share of query rows whose true item is in the top ``k``.

    Each query is a ``(user_id, time, true_news_id)`` tuple. An empty query
    iterable has undefined recall and returns NaN.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    n_queries = 0
    n_hits = 0
    for user_id, time, true_news_id in queries:
        retrieved = retrieve_fn(user_id, time, k)
        n_hits += true_news_id in retrieved[:k]
        n_queries += 1

    return n_hits / n_queries if n_queries else float("nan")
