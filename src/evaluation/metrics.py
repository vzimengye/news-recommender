"""Ranking and retrieval metrics for news recommendation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from math import log2
from typing import Any

import numpy as np
import pandas as pd


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

    Vectorized across all impressions at once (grouped rank + grouped sum/min)
    instead of looping per impression: AUC is the Mann-Whitney rank-sum
    statistic, which is exactly equivalent to ROC AUC including tie handling,
    so this needs no per-impression calls into sklearn. This matters at MIND
    scale -- a Python-level loop calling roc_auc_score once per impression
    does not finish on the 73k-impression dev split in a reasonable time.
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

    frame = pd.DataFrame(
        {
            "impression_id": df["impression_id"].to_numpy(),
            "label": labels.to_numpy(dtype=int),
            "score": scores.to_numpy(dtype=float),
        }
    )
    n_impressions = int(frame["impression_id"].nunique(dropna=False))

    grouped_score = frame.groupby("impression_id", sort=False)["score"]
    # Average rank (ties split credit) is what the AUC rank-sum formula needs.
    frame["asc_rank"] = grouped_score.rank(method="average")
    # "first" (ties broken by row order) matches the old stable-argsort
    # tie-break for MRR/nDCG's best-ranked-click lookup.
    frame["desc_rank"] = grouped_score.rank(method="first", ascending=False)

    grouped = frame.groupby("impression_id", sort=False)
    n_pos = grouped["label"].sum()
    n_neg = grouped.size() - n_pos
    eligible = (n_pos > 0) & (n_neg > 0)
    n_skipped = int((~eligible).sum())

    if not eligible.any():
        nan = float("nan")
        return {
            "auc": nan,
            "mrr": nan,
            "ndcg@5": nan,
            "ndcg@10": nan,
            "n_impressions": n_impressions,
            "n_skipped": n_skipped,
        }

    frame["eligible"] = frame["impression_id"].map(eligible)
    n_pos_elig = n_pos.loc[eligible]
    n_neg_elig = n_neg.loc[eligible]

    pos_rows = frame.loc[frame["eligible"] & (frame["label"] == 1)]
    pos_grouped = pos_rows.groupby("impression_id", sort=False)

    # AUC via the Mann-Whitney rank-sum statistic: among positive items'
    # average ranks, subtract the rank-sum an all-tied group would get, then
    # normalize by the number of positive/negative pairs.
    rank_sum_pos = pos_grouped["asc_rank"].sum()
    auc_per_impression = (
        rank_sum_pos - n_pos_elig * (n_pos_elig + 1) / 2
    ) / (n_pos_elig * n_neg_elig)

    best_desc_rank = pos_grouped["desc_rank"].min()
    mrr_per_impression = 1.0 / best_desc_rank

    def mean_ndcg_at_k(k: int) -> float:
        in_top_k = pos_rows.loc[pos_rows["desc_rank"] <= k]
        contribution = 1.0 / np.log2(in_top_k["desc_rank"] + 1)
        dcg = (
            contribution.groupby(in_top_k["impression_id"], sort=False)
            .sum()
            .reindex(n_pos_elig.index, fill_value=0.0)
        )
        idcg_lookup = {
            n: _dcg_at_k([1] * n, k) for n in n_pos_elig.unique().tolist()
        }
        idcg = n_pos_elig.map(idcg_lookup)
        return float((dcg / idcg).mean())

    return {
        "auc": float(auc_per_impression.mean()),
        "mrr": float(mrr_per_impression.mean()),
        "ndcg@5": mean_ndcg_at_k(5),
        "ndcg@10": mean_ndcg_at_k(10),
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
