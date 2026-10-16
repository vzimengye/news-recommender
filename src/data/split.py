"""Time-aware statistics and holdout splitting for impression samples."""

from __future__ import annotations

from math import ceil
from numbers import Integral, Real

import pandas as pd


def daily_exposure_stats(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarize daily impressions, candidate rows, clicks, and click rate.

    ``samples`` is candidate-level data: every row represents one candidate
    shown in an impression. Impression counts are therefore deduplicated by
    ``impression_id`` while candidate counts and click-through rate use rows.
    """
    required = {"time", "impression_id", "label"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples is missing required columns: {sorted(missing)}")

    if samples.empty:
        return pd.DataFrame(
            columns=["date", "impressions", "candidate_samples", "clicks", "ctr"]
        )

    timestamps = pd.to_datetime(samples["time"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("samples contains missing or invalid timestamps")

    daily = samples.assign(date=timestamps.dt.date).groupby("date", sort=True).agg(
        impressions=("impression_id", "nunique"),
        candidate_samples=("impression_id", "size"),
        clicks=("label", "sum"),
    )
    daily["ctr"] = daily["clicks"] / daily["candidate_samples"]
    return daily.reset_index()


def time_ordered_holdout(
    samples: pd.DataFrame,
    *,
    holdout_days: int | None = None,
    holdout_fraction: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split candidate samples into earlier train and later holdout portions.

    Specify exactly one of ``holdout_days`` or ``holdout_fraction``. The day
    option selects the final rolling ``N * 24`` hours relative to the latest
    timestamp. The fraction option selects the latest fraction of distinct
    impressions. All candidate rows from an impression stay together, and
    impressions sharing the cutoff timestamp stay on the same side to avoid
    splitting tied times.

    The function returns ``(train_part, holdout_part)`` in their original row
    order. It does not write files or create a third validation dataset.
    """
    if (holdout_days is None) == (holdout_fraction is None):
        raise ValueError("specify exactly one of holdout_days or holdout_fraction")
    if holdout_days is not None and (
        isinstance(holdout_days, bool)
        or not isinstance(holdout_days, Integral)
        or holdout_days <= 0
    ):
        raise ValueError("holdout_days must be a positive integer")
    if holdout_fraction is not None:
        if not isinstance(holdout_fraction, Real) or not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between 0 and 1, exclusive")

    required = {"impression_id", "time"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples is missing required columns: {sorted(missing)}")
    if samples.empty:
        raise ValueError("cannot split an empty samples DataFrame")

    timestamps = pd.to_datetime(samples["time"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("samples contains missing or invalid timestamps")
    if samples["impression_id"].isna().any():
        raise ValueError("samples contains missing impression_id values")

    impression_times = pd.DataFrame(
        {"impression_id": samples["impression_id"], "time": timestamps}
    )
    times_per_impression = impression_times.groupby("impression_id", sort=False)[
        "time"
    ].nunique()
    if (times_per_impression != 1).any():
        raise ValueError("each impression_id must have exactly one timestamp")
    impressions = impression_times.drop_duplicates("impression_id").sort_values("time")

    if holdout_days is not None:
        cutoff = timestamps.max() - pd.Timedelta(days=holdout_days)
    else:
        assert holdout_fraction is not None
        holdout_count = max(1, ceil(len(impressions) * holdout_fraction))
        cutoff = impressions.iloc[-holdout_count]["time"]

    holdout_mask = timestamps >= cutoff
    if not holdout_mask.any() or holdout_mask.all():
        raise ValueError(
            "the requested holdout leaves an empty train or holdout partition; "
            "choose a smaller window/fraction or inspect timestamp ties"
        )

    train_part = samples.loc[~holdout_mask].copy()
    holdout_part = samples.loc[holdout_mask].copy()
    return train_part, holdout_part
