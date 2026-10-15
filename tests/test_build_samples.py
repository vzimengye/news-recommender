"""Tests for src.data.build_samples."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.build_samples import build_samples
from src.data.parse import parse_behaviors, parse_news

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture_frames() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    train_behaviors = parse_behaviors(FIXTURES / "behaviors_train.tsv")
    dev_behaviors = parse_behaviors(FIXTURES / "behaviors_dev.tsv")
    train_news = parse_news(FIXTURES / "news_train.tsv")
    dev_news = parse_news(FIXTURES / "news_dev.tsv")
    return train_behaviors, dev_behaviors, train_news, dev_news


def test_build_samples_expands_one_row_per_candidate(tmp_path: Path) -> None:
    train_behaviors, dev_behaviors, train_news, dev_news = _load_fixture_frames()

    paths = build_samples(
        train_behaviors, dev_behaviors, train_news, dev_news, output_dir=tmp_path
    )
    train_samples = pd.read_parquet(paths["train_samples"])
    dev_samples = pd.read_parquet(paths["dev_samples"])

    # Impression 1 and impression 2 each show two candidates.
    assert len(train_samples) == 4
    # Impression 3 shows two candidates.
    assert len(dev_samples) == 2

    assert list(train_samples.columns) == [
        "impression_id",
        "user_id",
        "time",
        "news_id",
        "label",
        "history",
    ]

    impression_1 = train_samples[train_samples["impression_id"] == "1"]
    labels_by_news = dict(zip(impression_1["news_id"], impression_1["label"]))
    assert labels_by_news == {"N1": 0, "N2": 1}
    assert all(list(h) == [] for h in impression_1["history"])

    impression_2 = train_samples[train_samples["impression_id"] == "2"]
    labels_by_news = dict(zip(impression_2["news_id"], impression_2["label"]))
    assert labels_by_news == {"N2": 0, "N3": 1}
    assert all(list(h) == ["N1"] for h in impression_2["history"])


def test_build_samples_first_seen_is_earliest_across_splits(tmp_path: Path) -> None:
    train_behaviors, dev_behaviors, train_news, dev_news = _load_fixture_frames()

    paths = build_samples(
        train_behaviors, dev_behaviors, train_news, dev_news, output_dir=tmp_path
    )
    first_seen = pd.read_parquet(paths["news"]).set_index("news_id")["first_seen"]

    # N1 and N2 first appear in train impression 1 (8:00 AM).
    assert first_seen["N1"] == datetime(2019, 11, 10, 8, 0, 0)
    assert first_seen["N2"] == datetime(2019, 11, 10, 8, 0, 0)
    # N3 appears in train impression 2 (9:00 AM), earlier than dev impression
    # 3 (10:00 AM) -- the earlier, train-side sighting must win.
    assert first_seen["N3"] == datetime(2019, 11, 10, 9, 0, 0)
    # N4 only appears in dev impression 3.
    assert first_seen["N4"] == datetime(2019, 11, 12, 10, 0, 0)


def test_build_samples_news_table_is_deduplicated_across_splits(
    tmp_path: Path,
) -> None:
    train_behaviors, dev_behaviors, train_news, dev_news = _load_fixture_frames()

    paths = build_samples(
        train_behaviors, dev_behaviors, train_news, dev_news, output_dir=tmp_path
    )
    news = pd.read_parquet(paths["news"])

    assert sorted(news["news_id"]) == ["N1", "N2", "N3", "N4"]
    assert news["news_id"].is_unique
