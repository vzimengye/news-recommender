"""Shared index of per-news click timestamps, built once from train.

Both retrieval (src/retrieval/popular.py) and feature building use this same
index so there is exactly one place that defines "when was this news item
clicked" -- the structure every causal, windowed-count feature or retriever
in this project is built from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_click_time_index(train_samples: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map each news ID to its sorted array of click timestamps in train.

    ``train_samples`` needs ``news_id``, ``time``, ``label`` columns. News
    IDs that were never clicked are simply absent from the returned dict.
    """
    clicks = train_samples.loc[train_samples["label"] == 1, ["news_id", "time"]]
    return {
        news_id: np.sort(pd.to_datetime(group["time"]).to_numpy())
        for news_id, group in clicks.groupby("news_id", sort=False)
    }
