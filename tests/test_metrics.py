"""Tests for src.evaluation.metrics."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.evaluation.metrics import ranking_metrics, recall_at_k


def test_ranking_metrics_matches_hand_computed_single_impression() -> None:
    frame = pd.DataFrame(
        {
            "impression_id": ["imp-1"] * 5,
            "label": [0, 0, 1, 0, 0],
            "score": [0.1, 0.2, 0.9, 0.3, 0.4],
        }
    )

    # Descending scores rank the clicked item first. AUC = 4 winning positive-
    # negative pairs / 4 pairs = 1. MRR = 1/1 = 1. For both cutoffs, DCG is
    # (2^1 - 1)/log2(1 + 1) = 1, and IDCG is also 1, so nDCG@5 = nDCG@10 = 1.
    metrics = ranking_metrics(frame)

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["ndcg@5"] == pytest.approx(1.0)
    assert metrics["ndcg@10"] == pytest.approx(1.0)
    assert metrics["n_impressions"] == 1
    assert metrics["n_skipped"] == 0


def test_ranking_metrics_matches_hand_computed_imperfect_ranking() -> None:
    frame = pd.DataFrame(
        {
            "impression_id": ["imp-1"] * 5,
            "label": [0, 0, 0, 1, 0],
            "score": [0.9, 0.5, 0.3, 0.4, 0.1],
        }
    )

    # Sorted by score descending: 0.9(0), 0.5(0), 0.4(1), 0.3(0), 0.1(0) --
    # the click lands at rank 3, not rank 1, so this (unlike the perfect-
    # ranking case above) actually exercises the rank-dependent formulas.
    # AUC: the positive (score 0.4) beats negatives 0.3 and 0.1 but loses to
    #   0.9 and 0.5, so 2 winning pairs / 4 total pairs = 0.5.
    # MRR: the click is the 3rd-ranked item, so MRR = 1/3.
    # DCG@5 = (2^1 - 1)/log2(3 + 1) = 1/2 = 0.5. The ideal ranking puts the
    #   click first, giving IDCG@5 = (2^1 - 1)/log2(1 + 1) = 1. So
    #   nDCG@5 = 0.5/1 = 0.5. There are only 5 candidates, so nDCG@10 is the
    #   same ranking truncated to 5 items and equals nDCG@5.
    metrics = ranking_metrics(frame)

    assert metrics["auc"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["ndcg@5"] == pytest.approx(0.5)
    assert metrics["ndcg@10"] == pytest.approx(0.5)


def test_ranking_metrics_skips_all_zero_and_all_one_impressions() -> None:
    frame = pd.DataFrame(
        {
            "impression_id": ["valid"] * 2 + ["all-zero"] * 2 + ["all-one"] * 2,
            "label": [0, 1, 0, 0, 1, 1],
            "score": [0.1, 0.9, 0.2, 0.3, 0.8, 0.7],
        }
    )

    metrics = ranking_metrics(frame)

    assert metrics["n_impressions"] == 3
    assert metrics["n_skipped"] == 2
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["ndcg@5"] == pytest.approx(1.0)
    assert metrics["ndcg@10"] == pytest.approx(1.0)


def test_ranking_metrics_returns_nan_when_all_impressions_are_skipped() -> None:
    frame = pd.DataFrame(
        {"impression_id": ["zero", "one"], "label": [0, 1], "score": [0.2, 0.8]}
    )

    metrics = ranking_metrics(frame)

    assert metrics["n_impressions"] == 2
    assert metrics["n_skipped"] == 2
    assert pd.isna(metrics["auc"])
    assert pd.isna(metrics["mrr"])
    assert pd.isna(metrics["ndcg@5"])
    assert pd.isna(metrics["ndcg@10"])


def test_recall_at_k_always_hit_never_hit_and_partial_hit() -> None:
    queries = [
        ("user-1", datetime(2024, 1, 1), "news-1"),
        ("user-2", datetime(2024, 1, 2), "news-2"),
        ("user-3", datetime(2024, 1, 3), "news-3"),
    ]

    def always_hit(user_id: str, time: datetime, k: int) -> list[str]:
        true_id = next(news_id for uid, _, news_id in queries if uid == user_id)
        return [true_id]

    def never_hit(user_id: str, time: datetime, k: int) -> list[str]:
        return ["not-the-answer"]

    partial_results = {"user-1": ["news-1"], "user-2": ["news-2"], "user-3": ["other"]}

    def partial_hit(user_id: str, time: datetime, k: int) -> list[str]:
        return partial_results[user_id]

    assert recall_at_k(queries, always_hit, k=1) == pytest.approx(1.0)
    assert recall_at_k(queries, never_hit, k=1) == pytest.approx(0.0)
    # Two of the three query targets are retrieved: recall@1 = 2 / 3.
    assert recall_at_k(queries, partial_hit, k=1) == pytest.approx(2 / 3)
