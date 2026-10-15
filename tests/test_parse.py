"""Tests for src.data.parse."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.data.parse import parse_behaviors, parse_news

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_behaviors_reads_expected_fields() -> None:
    behaviors = parse_behaviors(FIXTURES / "behaviors_train.tsv")

    assert list(behaviors.columns) == [
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions",
    ]
    assert len(behaviors) == 2

    first, second = behaviors.iloc[0], behaviors.iloc[1]
    assert first["impression_id"] == "1"
    assert first["user_id"] == "U1"
    assert first["time"] == datetime(2019, 11, 10, 8, 0, 0)
    assert first["history"] == []
    assert first["impressions"] == [("N1", 0), ("N2", 1)]

    assert second["history"] == ["N1"]
    assert second["impressions"] == [("N2", 0), ("N3", 1)]


def test_parse_behaviors_rejects_wrong_field_count(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_behaviors.tsv"
    bad_file.write_text("1\tU1\t11/10/2019 8:00:00 AM\tN1-0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 5 tab-separated fields"):
        parse_behaviors(bad_file)


def test_parse_behaviors_rejects_invalid_timestamp(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_behaviors.tsv"
    bad_file.write_text("1\tU1\tnot-a-time\t\tN1-0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid MIND timestamp"):
        parse_behaviors(bad_file)


def test_parse_behaviors_rejects_impression_token_without_label(
    tmp_path: Path,
) -> None:
    bad_file = tmp_path / "bad_behaviors.tsv"
    bad_file.write_text("1\tU1\t11/10/2019 8:00:00 AM\t\tN1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed impression"):
        parse_behaviors(bad_file)


def test_parse_behaviors_rejects_impression_token_with_bad_label(
    tmp_path: Path,
) -> None:
    bad_file = tmp_path / "bad_behaviors.tsv"
    bad_file.write_text("1\tU1\t11/10/2019 8:00:00 AM\t\tN1-2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected news_id-0 or news_id-1"):
        parse_behaviors(bad_file)


def test_parse_news_keeps_only_the_first_six_columns() -> None:
    news = parse_news(FIXTURES / "news_train.tsv")

    assert list(news.columns) == [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
    ]
    assert len(news) == 2
    assert news.iloc[0]["news_id"] == "N1"
    assert news.iloc[0]["category"] == "news"
    assert news.iloc[0]["title"] == "Title One"


def test_parse_news_rejects_too_few_fields(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_news.tsv"
    bad_file.write_text("N1\tnews\tpolitics\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected at least 6"):
        parse_news(bad_file)
