"""Candidate-level feature construction, shared by training and serving.

Every feature here is causal: it only ever uses information available
strictly before the candidate's impression time ``t``. Popularity counts
come from an index built once from train and passed in (see
``click_index.build_click_time_index``) rather than rebuilt per call, so
training and serving read the exact same numbers for the same inputs.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

DEFAULT_POPULARITY_WINDOWS_HOURS: tuple[float, ...] = (1.0, 6.0, 24.0)


def _windowed_click_counts(
    samples: pd.DataFrame,
    click_index: dict[str, np.ndarray],
    windows_hours: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """For each row's (news_id, time), count clicks strictly before time.

    Grouped by news_id (tens of thousands of candidates) rather than looped
    row by row (millions of rows): each group does one vectorized
    ``searchsorted`` call per window instead of one Python-level bisect per
    row, which is the difference between seconds and hours at train scale.
    """
    frame = pd.DataFrame(
        {
            "news_id": samples["news_id"].to_numpy(),
            "time": pd.to_datetime(samples["time"]).to_numpy(),
            "_pos": np.arange(len(samples)),
        }
    )
    results = {w: np.zeros(len(samples), dtype=np.int32) for w in windows_hours}

    for news_id, group in frame.groupby("news_id", sort=False):
        times = click_index.get(news_id)
        if times is None:
            continue
        query_times = group["time"].to_numpy()
        pos = group["_pos"].to_numpy()

        # "Before t" is a strict cutoff; searchsorted(side="right") would
        # include a click exactly at t, so that count is found once per
        # group and subtracted below for every window.
        hi = times.searchsorted(query_times, side="right")
        hi_clipped = np.clip(hi - 1, 0, len(times) - 1)
        exact_at_t = (hi > 0) & (times[hi_clipped] == query_times)

        for window_hours in windows_hours:
            window_start = query_times - np.timedelta64(
                int(window_hours * 3600), "s"
            )
            lo = times.searchsorted(window_start, side="left")
            counts = (hi - lo) - (exact_at_t & (hi > lo)).astype(np.int32)
            results[window_hours][pos] = counts

    return results


def _history_category_shares(
    samples: pd.DataFrame, news_category: dict[str, str]
) -> dict[str, dict[str, float]]:
    """Per impression, the share of history items in each category.

    History is identical for every candidate row of the same impression, so
    this is computed once per impression (hundreds of thousands) rather than
    once per candidate row (millions).
    """
    # Plain-dict Counter, not a pandas Series + value_counts() per impression:
    # constructing a pandas object per impression (hundreds of thousands of
    # them) dominated runtime (~40 minutes on train) despite each call being
    # individually "small" -- Counter has none of that per-call overhead.
    per_impression = samples.drop_duplicates("impression_id")[
        ["impression_id", "history"]
    ]
    shares: dict[str, dict[str, float]] = {}
    for impression_id, history in zip(
        per_impression["impression_id"], per_impression["history"]
    ):
        categories = [news_category[nid] for nid in history if nid in news_category]
        if not categories:
            shares[impression_id] = {}
            continue
        counts = Counter(categories)
        total = len(categories)
        shares[impression_id] = {cat: n / total for cat, n in counts.items()}
    return shares


def build_features(
    samples: pd.DataFrame,
    news: pd.DataFrame,
    click_index: dict[str, np.ndarray],
    *,
    popularity_windows_hours: tuple[float, ...] = DEFAULT_POPULARITY_WINDOWS_HOURS,
) -> pd.DataFrame:
    """Build one feature row per candidate in ``samples``.

    ``samples`` needs ``impression_id``, ``time``, ``news_id``, ``history``
    (list of news IDs). ``news`` needs ``news_id``, ``category``,
    ``subcategory``, ``first_seen``. ``click_index`` is
    ``click_index.build_click_time_index(train_samples)``, built once from
    train and reused for every call (training and serving alike).

    Returns a DataFrame aligned with ``samples``' row order and index, with
    one ``pop_{H}h`` column per window in ``popularity_windows_hours``, plus
    ``freshness_hours``, ``category_affinity``, ``subcategory_affinity``, and
    ``history_length``.
    """
    required_samples = {"impression_id", "time", "news_id", "history"}
    missing = required_samples.difference(samples.columns)
    if missing:
        raise ValueError(f"samples is missing required columns: {sorted(missing)}")
    required_news = {"news_id", "category", "subcategory", "first_seen"}
    missing = required_news.difference(news.columns)
    if missing:
        raise ValueError(f"news is missing required columns: {sorted(missing)}")

    pop_counts = _windowed_click_counts(samples, click_index, popularity_windows_hours)

    first_seen = dict(zip(news["news_id"], pd.to_datetime(news["first_seen"])))
    candidate_time = pd.to_datetime(samples["time"])
    candidate_first_seen = samples["news_id"].map(first_seen)
    freshness_hours = (
        candidate_time - candidate_first_seen
    ).dt.total_seconds() / 3600.0

    news_category = dict(zip(news["news_id"], news["category"]))
    news_subcategory = dict(zip(news["news_id"], news["subcategory"]))
    category_shares = _history_category_shares(samples, news_category)
    subcategory_shares = _history_category_shares(samples, news_subcategory)

    candidate_category = samples["news_id"].map(news_category)
    candidate_subcategory = samples["news_id"].map(news_subcategory)
    category_affinity = [
        category_shares.get(impression_id, {}).get(category, 0.0)
        for impression_id, category in zip(samples["impression_id"], candidate_category)
    ]
    subcategory_affinity = [
        subcategory_shares.get(impression_id, {}).get(subcategory, 0.0)
        for impression_id, subcategory in zip(
            samples["impression_id"], candidate_subcategory
        )
    ]

    features = pd.DataFrame(index=samples.index)
    for window_hours in popularity_windows_hours:
        label = f"pop_{window_hours:g}h"
        features[label] = pop_counts[window_hours]
    features["freshness_hours"] = freshness_hours.to_numpy()
    features["category_affinity"] = category_affinity
    features["subcategory_affinity"] = subcategory_affinity
    features["history_length"] = samples["history"].map(len).to_numpy()

    return features
