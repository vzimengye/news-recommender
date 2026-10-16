"""Tests for src.data.split."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data.split import daily_exposure_stats, time_ordered_holdout


def _samples(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    """Build a candidate-level samples frame from (impression_id, time, label)."""
    return pd.DataFrame(rows, columns=["impression_id", "time", "label"])


def test_daily_exposure_stats_aggregates_by_day() -> None:
    samples = _samples(
        [
            ("1", "2024-01-01 08:00:00", 1),
            ("1", "2024-01-01 08:00:00", 0),
            ("2", "2024-01-01 09:00:00", 0),
            ("2", "2024-01-01 09:00:00", 0),
            ("3", "2024-01-02 08:00:00", 1),
            ("3", "2024-01-02 08:00:00", 1),
        ]
    )

    daily = daily_exposure_stats(samples).set_index("date")

    day1 = daily.loc[pd.Timestamp("2024-01-01").date()]
    assert day1["impressions"] == 2
    assert day1["candidate_samples"] == 4
    assert day1["clicks"] == 1
    assert day1["ctr"] == pytest.approx(0.25)

    day2 = daily.loc[pd.Timestamp("2024-01-02").date()]
    assert day2["impressions"] == 1
    assert day2["candidate_samples"] == 2
    assert day2["clicks"] == 2
    assert day2["ctr"] == pytest.approx(1.0)


def test_daily_exposure_stats_rejects_missing_columns() -> None:
    samples = pd.DataFrame({"impression_id": ["1"], "time": ["2024-01-01 08:00:00"]})

    with pytest.raises(ValueError, match="missing required columns"):
        daily_exposure_stats(samples)


def test_daily_exposure_stats_rejects_invalid_timestamps() -> None:
    samples = _samples([("1", "not-a-time", 0)])

    with pytest.raises(ValueError, match="invalid timestamps"):
        daily_exposure_stats(samples)


def test_time_ordered_holdout_requires_exactly_one_option() -> None:
    samples = _samples([("1", "2024-01-01 08:00:00", 0)])

    with pytest.raises(ValueError, match="specify exactly one"):
        time_ordered_holdout(samples)
    with pytest.raises(ValueError, match="specify exactly one"):
        time_ordered_holdout(samples, holdout_days=1, holdout_fraction=0.1)


def test_time_ordered_holdout_rejects_invalid_holdout_days() -> None:
    samples = _samples([("1", "2024-01-01 08:00:00", 0)])

    with pytest.raises(ValueError, match="holdout_days must be a positive integer"):
        time_ordered_holdout(samples, holdout_days=0)
    with pytest.raises(ValueError, match="holdout_days must be a positive integer"):
        time_ordered_holdout(samples, holdout_days=True)  # bool is an Integral


def test_time_ordered_holdout_rejects_invalid_holdout_fraction() -> None:
    samples = _samples([("1", "2024-01-01 08:00:00", 0)])

    with pytest.raises(ValueError, match="holdout_fraction must be between 0 and 1"):
        time_ordered_holdout(samples, holdout_fraction=0.0)
    with pytest.raises(ValueError, match="holdout_fraction must be between 0 and 1"):
        time_ordered_holdout(samples, holdout_fraction=1.5)


def test_time_ordered_holdout_rejects_inconsistent_impression_timestamps() -> None:
    samples = _samples(
        [
            ("1", "2024-01-01 08:00:00", 0),
            ("1", "2024-01-01 09:00:00", 1),
        ]
    )

    with pytest.raises(ValueError, match="exactly one timestamp"):
        time_ordered_holdout(samples, holdout_fraction=0.5)


def test_time_ordered_holdout_rejects_empty_frame() -> None:
    samples = _samples([])

    with pytest.raises(ValueError, match="cannot split an empty"):
        time_ordered_holdout(samples, holdout_fraction=0.5)


def test_time_ordered_holdout_by_days_keeps_train_strictly_before_holdout() -> None:
    samples = _samples(
        [
            ("1", "2024-01-01 08:00:00", 0),
            ("1", "2024-01-01 08:00:00", 1),
            ("2", "2024-01-02 08:00:00", 0),
            ("2", "2024-01-02 08:00:00", 0),
            ("3", "2024-01-03 08:00:00", 1),
            ("3", "2024-01-03 08:00:00", 0),
        ]
    )

    train, holdout = time_ordered_holdout(samples, holdout_days=1)

    assert set(train["impression_id"]) == {"1"}
    assert set(holdout["impression_id"]) == {"2", "3"}
    assert pd.to_datetime(train["time"]).max() < pd.to_datetime(holdout["time"]).min()
    # No row is duplicated or dropped across the two partitions.
    assert len(train) + len(holdout) == len(samples)


def test_time_ordered_holdout_by_fraction_keeps_tied_impressions_together() -> None:
    samples = _samples(
        [
            ("1", "2024-01-01 09:00:00", 0),
            ("2", "2024-01-01 10:00:00", 0),
            ("3", "2024-01-01 11:00:00", 0),
            ("4", "2024-01-01 12:00:00", 0),
            ("5", "2024-01-01 13:00:00", 1),
            ("6", "2024-01-01 13:00:00", 0),
        ]
    )

    # Requesting 1/6 only asks for impression 6 (the last by time), but
    # impression 5 shares its exact timestamp and must not be split off.
    train, holdout = time_ordered_holdout(samples, holdout_fraction=1 / 6)

    assert set(holdout["impression_id"]) == {"5", "6"}
    assert set(train["impression_id"]) == {"1", "2", "3", "4"}
