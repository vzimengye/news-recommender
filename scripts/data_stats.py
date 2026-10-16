"""Print basic counts and cold-start statistics for processed MIND-small data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq


@dataclass
class SplitStats:
    """Streaming statistics for one candidate-level samples file."""

    sample_count: int = 0
    clicks: int = 0
    users: set[str] = field(default_factory=set)
    impressions: set[str] = field(default_factory=set)
    candidate_news: set[str] = field(default_factory=set)
    clicked_news: set[str] = field(default_factory=set)
    daily: dict[date, "DailyStats"] = field(default_factory=dict)


@dataclass
class DailyStats:
    impressions: set[str] = field(default_factory=set)
    candidate_samples: int = 0
    clicks: int = 0


def _collect_stats(path: Path) -> SplitStats:
    """Read only the columns needed, in batches to limit memory use."""
    parquet = pq.ParquetFile(path)
    stats = SplitStats(sample_count=parquet.metadata.num_rows)

    required = {"impression_id", "user_id", "news_id", "label", "time"}
    available = set(parquet.schema_arrow.names)
    missing = required - available
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=["impression_id", "user_id", "news_id", "label", "time"],
    ):
        impression_ids = batch.column("impression_id").to_pylist()
        stats.impressions.update(impression_ids)
        stats.users.update(batch.column("user_id").to_pylist())

        news_ids = batch.column("news_id").to_pylist()
        labels = batch.column("label").to_pylist()
        times = batch.column("time").to_pylist()
        stats.candidate_news.update(news_ids)
        stats.clicks += sum(labels)
        stats.clicked_news.update(
            news_id for news_id, label in zip(news_ids, labels) if label == 1
        )
        for impression_id, label, timestamp in zip(impression_ids, labels, times):
            day_stats = stats.daily.setdefault(timestamp.date(), DailyStats())
            day_stats.impressions.add(impression_id)
            day_stats.candidate_samples += 1
            day_stats.clicks += label

    return stats


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.2%}"


def _print_split(name: str, stats: SplitStats) -> None:
    impression_count = len(stats.impressions)
    ctr = _percent(stats.clicks, stats.sample_count)
    avg_candidates = (
        stats.sample_count / impression_count if impression_count else 0.0
    )

    print(f"{name}:")
    print(f"  unique users:                 {len(stats.users):,}")
    print(f"  unique candidate news:        {len(stats.candidate_news):,}")
    print(f"  impressions:                  {impression_count:,}")
    print(f"  expanded candidate samples:  {stats.sample_count:,}")
    print(f"  clicks:                       {stats.clicks:,}")
    print(f"  click-through rate:           {ctr}")
    print(f"  avg candidates / impression:  {avg_candidates:.2f}")
    print("  daily exposure distribution:")
    for day, daily in sorted(stats.daily.items()):
        daily_ctr = _percent(daily.clicks, daily.candidate_samples)
        print(
            f"    {day.isoformat()}: "
            f"{len(daily.impressions):,} impressions, "
            f"{daily.candidate_samples:,} candidates, "
            f"{daily.clicks:,} clicks, CTR {daily_ctr}"
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Print train/dev counts and news cold-start shares."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=repo_root / "data" / "processed",
        help="Directory containing train_samples.parquet, dev_samples.parquet, "
        "and news.parquet (default: data/processed).",
    )
    args = parser.parse_args()

    train_path = args.processed_dir / "train_samples.parquet"
    dev_path = args.processed_dir / "dev_samples.parquet"
    news_path = args.processed_dir / "news.parquet"
    missing_paths = [
        path for path in (train_path, dev_path, news_path) if not path.is_file()
    ]
    if missing_paths:
        expected = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing processed data: {expected}. Run the data parsing/sample "
            "builder first."
        )

    train = _collect_stats(train_path)
    dev = _collect_stats(dev_path)
    news_count = pq.ParquetFile(news_path).metadata.num_rows

    _print_split("Train", train)
    _print_split("Dev", dev)
    print(f"News metadata rows: {news_count:,}")

    dev_unseen_exposure = dev.candidate_news - train.candidate_news
    dev_unseen_click = dev.candidate_news - train.clicked_news

    # Candidate-level rates are weighted by how often each article appeared;
    # unique-news rates count each news ID once. Item-CF's true cold-start pool
    # is the set with no positive train interaction, not merely no train exposure.
    train_candidate_news = train.candidate_news
    train_clicked_news = train.clicked_news
    unseen_exposure_count = 0
    unseen_click_count = 0
    for batch in pq.ParquetFile(dev_path).iter_batches(
        batch_size=250_000, columns=["news_id"]
    ):
        news_ids = batch.column("news_id").to_pylist()
        unseen_exposure_count += sum(
            news_id not in train_candidate_news for news_id in news_ids
        )
        unseen_click_count += sum(
            news_id not in train_clicked_news for news_id in news_ids
        )

    print("Dev news absent from train:")
    print(
        "  unique news never exposed in train: "
        f"{len(dev_unseen_exposure):,}/{len(dev.candidate_news):,} "
        f"({_percent(len(dev_unseen_exposure), len(dev.candidate_news))})"
    )
    print(
        "  candidate impressions with no train exposure: "
        f"{unseen_exposure_count:,}/{dev.sample_count:,} "
        f"({_percent(unseen_exposure_count, dev.sample_count)})"
    )
    print(
        "  unique news with no positive train click (item-CF cold start): "
        f"{len(dev_unseen_click):,}/{len(dev.candidate_news):,} "
        f"({_percent(len(dev_unseen_click), len(dev.candidate_news))})"
    )
    print(
        "  candidate impressions with no positive train click: "
        f"{unseen_click_count:,}/{dev.sample_count:,} "
        f"({_percent(unseen_click_count, dev.sample_count)})"
    )


if __name__ == "__main__":
    main()
