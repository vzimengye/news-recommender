"""Print ranking metrics for scorers on the MIND-small dev candidates."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.metrics import ranking_metrics  # noqa: E402

RANDOM_SEED = 0


def random_scores(dev: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Score candidates uniformly at random; a correct pipeline gives AUC ~0.5."""
    rng = np.random.default_rng(RANDOM_SEED)
    return pd.Series(rng.random(len(dev)), index=dev.index)


def global_popularity_scores(dev: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Score each candidate by its total click count in train (no time window)."""
    click_counts = train.groupby("news_id")["label"].sum()
    return dev["news_id"].map(click_counts).fillna(0.0)


SCORERS = {
    "random": random_scores,
    "global_popularity": global_popularity_scores,
}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Print impression-averaged ranking metrics for each "
        "registered scorer on the dev split."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=repo_root / "data" / "processed",
        help="Directory with train_samples.parquet and dev_samples.parquet "
        "(default: data/processed).",
    )
    args = parser.parse_args()

    train = pd.read_parquet(
        args.processed_dir / "train_samples.parquet",
        columns=["news_id", "label"],
    )
    dev = pd.read_parquet(
        args.processed_dir / "dev_samples.parquet",
        columns=["impression_id", "news_id", "label"],
    )

    header = f"{'scorer':<20}{'auc':>10}{'mrr':>10}{'ndcg@5':>10}{'ndcg@10':>10}{'n_impr':>10}{'n_skip':>10}"
    print(header)
    for name, scorer in SCORERS.items():
        t0 = time.time()
        scores = scorer(dev, train)
        metrics = ranking_metrics(
            dev.assign(score=scores)[["impression_id", "label", "score"]]
        )
        elapsed = time.time() - t0
        print(
            f"{name:<20}{metrics['auc']:>10.4f}{metrics['mrr']:>10.4f}"
            f"{metrics['ndcg@5']:>10.4f}{metrics['ndcg@10']:>10.4f}"
            f"{metrics['n_impressions']:>10}{metrics['n_skipped']:>10}"
            f"   ({elapsed:.1f}s)"
        )


if __name__ == "__main__":
    main()
