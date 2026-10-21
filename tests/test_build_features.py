"""Tests for src.features.build."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import build_features
from src.features.click_index import build_click_time_index


def _news(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["news_id", "category", "subcategory", "first_seen"]
    )


def test_popularity_features_never_count_clicks_at_or_after_impression_time() -> None:
    # A single candidate with one click clearly before t, one exactly at t,
    # and one clearly after t. Only the first should ever be counted.
    train = pd.DataFrame(
        {
            "news_id": ["N1", "N1", "N1"],
            "time": [
                "2024-01-02 08:30:00",  # before t: should count
                "2024-01-02 09:00:00",  # exactly at t: must not count
                "2024-01-02 09:30:00",  # after t: must not count (leakage)
            ],
            "label": [1, 1, 1],
        }
    )
    click_index = build_click_time_index(train)
    news = _news([("N1", "news", "politics", "2024-01-01 00:00:00")])
    samples = pd.DataFrame(
        {
            "impression_id": ["imp-1"],
            "time": ["2024-01-02 09:00:00"],
            "news_id": ["N1"],
            "history": [[]],
        }
    )

    features = build_features(samples, news, click_index)

    assert features.loc[0, "pop_1h"] == 1
    assert features.loc[0, "pop_6h"] == 1
    assert features.loc[0, "pop_24h"] == 1


def test_build_features_matches_hand_computed_values() -> None:
    train = pd.DataFrame(
        {
            "news_id": ["N1", "N1", "N1", "N1", "N1", "N3", "N3"],
            "time": [
                "2024-01-02 08:30:00",  # N1: in 1h/6h/24h windows
                "2024-01-02 02:00:00",  # N1: in 24h only (6h window starts 03:00)
                "2024-01-01 08:00:00",  # N1: outside even the 24h window
                "2024-01-02 09:00:00",  # N1: exactly at t, excluded
                "2024-01-02 15:00:00",  # N1: after t, excluded
                "2024-01-02 08:45:00",  # N3: in 1h/6h/24h windows
                "2024-01-02 09:30:00",  # N3: after t, excluded
            ],
            "label": [1, 1, 1, 1, 1, 1, 1],
        }
    )
    click_index = build_click_time_index(train)
    news = _news(
        [
            ("N1", "news", "politics", "2024-01-01 00:00:00"),
            ("N2", "sports", "nfl", "2023-01-01 00:00:00"),
            ("N3", "news", "breaking", "2024-01-01 12:00:00"),
            ("N4", "news", "elections", "2023-06-01 00:00:00"),
        ]
    )
    # History is half "sports/nfl" (N2), half "news/elections" (N4). Neither
    # candidate's own subcategory (politics, breaking) appears in history.
    samples = pd.DataFrame(
        {
            "impression_id": ["imp-1", "imp-1"],
            "time": ["2024-01-02 09:00:00", "2024-01-02 09:00:00"],
            "news_id": ["N1", "N3"],
            "history": [["N2", "N4"], ["N2", "N4"]],
        }
    )

    features = build_features(samples, news, click_index).set_index(
        samples["news_id"]
    )

    n1 = features.loc["N1"]
    assert n1["pop_1h"] == 1
    assert n1["pop_6h"] == 1
    assert n1["pop_24h"] == 2
    assert n1["freshness_hours"] == pytest.approx(33.0)
    assert n1["category_affinity"] == pytest.approx(0.5)
    assert n1["subcategory_affinity"] == pytest.approx(0.0)
    assert n1["history_length"] == 2

    n3 = features.loc["N3"]
    assert n3["pop_1h"] == 1
    assert n3["pop_6h"] == 1
    assert n3["pop_24h"] == 1
    assert n3["freshness_hours"] == pytest.approx(21.0)
    assert n3["category_affinity"] == pytest.approx(0.5)
    assert n3["subcategory_affinity"] == pytest.approx(0.0)
    assert n3["history_length"] == 2


def test_build_features_rejects_missing_columns() -> None:
    samples = pd.DataFrame(
        {"impression_id": ["imp-1"], "news_id": ["N1"], "history": [[]]}
    )
    news = _news([("N1", "news", "politics", "2024-01-01 00:00:00")])

    with pytest.raises(ValueError, match="samples is missing required columns"):
        build_features(samples, news, {})
