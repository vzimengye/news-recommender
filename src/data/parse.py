"""Parse the raw MIND TSV files into in-memory pandas DataFrames."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

import pandas as pd

PathLike: TypeAlias = str | Path

_BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
_NEWS_COLUMNS = ["news_id", "category", "subcategory", "title", "abstract", "url"]
_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def parse_behaviors(path: PathLike) -> pd.DataFrame:
    """Parse MIND ``behaviors.tsv``, keeping one row per impression.

    The returned columns are ``impression_id``, ``user_id``, ``time``,
    ``history``, and ``impressions``. ``time`` is datetime64, ``history`` is
    a list of news IDs, and ``impressions`` is a list of ``(news_id, label)``
    pairs. Candidate impressions are intentionally not expanded into rows.

    Raises:
        ValueError: If a row has the wrong number of fields, an invalid time,
            or a malformed candidate token.
    """
    rows: list[dict[str, object]] = []

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for line_number, fields in enumerate(reader, start=1):
            if len(fields) != 5:
                raise ValueError(
                    f"{path}: line {line_number}: expected 5 tab-separated "
                    f"fields, found {len(fields)}"
                )

            impression_id, user_id, raw_time, raw_history, raw_impressions = fields
            try:
                timestamp = datetime.strptime(raw_time, _TIME_FORMAT)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: line {line_number}: invalid MIND timestamp {raw_time!r}"
                ) from exc

            history = raw_history.split() if raw_history.strip() else []
            impressions: list[tuple[str, int]] = []
            for token in raw_impressions.split():
                try:
                    news_id, raw_label = token.rsplit("-", maxsplit=1)
                    label = int(raw_label)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{path}: line {line_number}: malformed impression {token!r}"
                    ) from exc

                if not news_id or label not in (0, 1):
                    raise ValueError(
                        f"{path}: line {line_number}: expected news_id-0 or "
                        f"news_id-1, found {token!r}"
                    )
                impressions.append((news_id, label))

            rows.append(
                {
                    "impression_id": impression_id,
                    "user_id": user_id,
                    "time": timestamp,
                    "history": history,
                    "impressions": impressions,
                }
            )

    return pd.DataFrame(rows, columns=_BEHAVIOR_COLUMNS)


def parse_news(path: PathLike) -> pd.DataFrame:
    """Parse MIND ``news.tsv``, retaining only the six requested columns.

    MIND news rows also contain entity annotation columns after ``url``;
    these are deliberately discarded here.

    Raises:
        ValueError: If a row has fewer than the six required fields.
    """
    rows: list[list[str]] = []

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for line_number, fields in enumerate(reader, start=1):
            if len(fields) < len(_NEWS_COLUMNS):
                raise ValueError(
                    f"{path}: line {line_number}: expected at least 6 tab-separated "
                    f"fields, found {len(fields)}"
                )
            rows.append(fields[: len(_NEWS_COLUMNS)])

    return pd.DataFrame(rows, columns=_NEWS_COLUMNS)
