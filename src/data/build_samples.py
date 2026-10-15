"""Build candidate-level samples and article first-seen timestamps."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import pandas as pd

PathLike: TypeAlias = str | Path

_SAMPLE_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "news_id",
    "label",
    "history",
]
_NEWS_COLUMNS = ["news_id", "category", "subcategory", "title", "abstract", "url"]


def _update_first_seen(
    first_seen: dict[str, pd.Timestamp], news_id: str, timestamp: pd.Timestamp
) -> None:
    """Record the earliest observed timestamp for an article ID."""
    previous = first_seen.get(news_id)
    if previous is None or timestamp < previous:
        first_seen[news_id] = timestamp


def _build_first_seen(
    behavior_frames: tuple[pd.DataFrame, ...],
) -> dict[str, pd.Timestamp]:
    """Find the earliest history or candidate occurrence across all splits."""
    first_seen: dict[str, pd.Timestamp] = {}

    for behaviors in behavior_frames:
        for row in behaviors.itertuples(index=False):
            timestamp = row.time
            for news_id in row.history:
                _update_first_seen(first_seen, news_id, timestamp)
            for news_id, _label in row.impressions:
                _update_first_seen(first_seen, news_id, timestamp)

    return first_seen


def _expand_candidates(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Expand each impression into one row per candidate article."""
    expanded = behaviors[
        ["impression_id", "user_id", "time", "history", "impressions"]
    ].explode("impressions", ignore_index=True)

    # explode represents an empty candidate list as NaN; it has no candidate
    # training examples, so remove it before extracting the tuple fields.
    expanded = expanded.loc[expanded["impressions"].notna()].copy()
    expanded["news_id"] = expanded["impressions"].str[0]
    expanded["label"] = expanded["impressions"].str[1].astype("int8")
    return expanded[_SAMPLE_COLUMNS]


def build_samples(
    train_behaviors: pd.DataFrame,
    dev_behaviors: pd.DataFrame,
    train_news: pd.DataFrame,
    dev_news: pd.DataFrame,
    output_dir: PathLike = "data/processed",
) -> dict[str, Path]:
    """Build and save train/dev candidate samples and the article table.

    ``first_seen`` is the minimum timestamp at which a news ID occurs in a
    behavior history or candidate list across both train and dev. This is
    fixed metadata: for a decision at time ``t``, compare it with ``t`` to
    determine whether the article had appeared by then.

    The function writes ``train_samples.parquet``, ``dev_samples.parquet``,
    and ``news.parquet`` under ``output_dir`` and returns their paths.
    It expects the parsed DataFrames produced by :mod:`src.data.parse`.
    """
    first_seen = _build_first_seen((train_behaviors, dev_behaviors))

    train_samples = _expand_candidates(train_behaviors)
    dev_samples = _expand_candidates(dev_behaviors)

    news = pd.concat(
        [train_news[_NEWS_COLUMNS], dev_news[_NEWS_COLUMNS]],
        ignore_index=True,
    ).drop_duplicates(subset="news_id", keep="first")
    news["first_seen"] = news["news_id"].map(first_seen)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_samples": output_path / "train_samples.parquet",
        "dev_samples": output_path / "dev_samples.parquet",
        "news": output_path / "news.parquet",
    }

    train_samples.to_parquet(paths["train_samples"], index=False)
    dev_samples.to_parquet(paths["dev_samples"], index=False)
    news.to_parquet(paths["news"], index=False)

    return paths
