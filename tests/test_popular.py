"""Tests for src.retrieval.popular."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.retrieval.popular import PopularityRetriever


def _train_samples(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["news_id", "time", "label"])


def _news(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["news_id", "first_seen"])


def test_retrieve_ranks_by_window_click_count_and_excludes_boundary_clicks() -> None:
    # Query at 2024-01-02 09:00:00 with the default 24h window looks back to
    # 2024-01-01 09:00:00 (inclusive) and excludes the query instant itself.
    train = _train_samples(
        [
            ("E", "2024-01-01 09:00:01", 1),
            ("E", "2024-01-01 09:00:02", 1),
            ("E", "2024-01-01 09:00:03", 1),
            ("A", "2024-01-01 20:00:00", 1),
            ("B", "2024-01-01 10:00:00", 1),
            ("B", "2024-01-01 10:00:00", 0),  # a non-click candidate, no effect
            ("F", "2024-01-02 09:00:00", 1),  # exactly at the cutoff: excluded
            ("G", "2024-01-01 08:59:59", 1),  # one second before window start: excluded
            ("H", "2024-01-01 23:00:00", 1),  # clicked, but not yet "available"
        ]
    )
    news = _news(
        [
            ("E", "2024-01-01 00:00:00"),
            ("A", "2024-01-01 00:00:00"),
            ("B", "2024-01-01 00:00:00"),
            ("F", "2024-01-01 00:00:00"),
            ("G", "2024-01-01 00:00:00"),
            ("H", "2024-01-02 10:00:00"),  # first_seen after the query time
        ]
    )
    retriever = PopularityRetriever(window_hours=24).fit(train, news)

    result = retriever.retrieve("any-user", datetime(2024, 1, 2, 9, 0, 0), k=3)

    assert result[0] == "E"
    assert set(result[1:3]) == {"A", "B"}
    assert "F" not in result
    assert "G" not in result
    assert "H" not in result


def test_retrieve_returns_fewer_than_k_when_not_enough_candidates() -> None:
    train = _train_samples([("A", "2024-01-01 08:00:00", 1)])
    news = _news([("A", "2024-01-01 00:00:00")])
    retriever = PopularityRetriever(window_hours=24).fit(train, news)

    result = retriever.retrieve("any-user", datetime(2024, 1, 1, 9, 0, 0), k=5)

    assert result == ["A"]


def test_retrieve_excludes_news_with_zero_clicks_in_window() -> None:
    train = _train_samples([("A", "2024-01-01 08:00:00", 0)])  # never clicked
    news = _news([("A", "2024-01-01 00:00:00")])
    retriever = PopularityRetriever(window_hours=24).fit(train, news)

    result = retriever.retrieve("any-user", datetime(2024, 1, 1, 9, 0, 0), k=5)

    assert result == []
