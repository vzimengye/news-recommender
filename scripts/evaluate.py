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
from src.features.build import build_features  # noqa: E402
from src.features.click_index import build_click_time_index  # noqa: E402
from src.ranking.lr import LogisticRegressionRanker  # noqa: E402

RANDOM_SEED = 0


def random_scores(dev: pd.DataFrame) -> pd.Series:
    """Score candidates uniformly at random; a correct pipeline gives AUC ~0.5."""
    rng = np.random.default_rng(RANDOM_SEED)
    return pd.Series(rng.random(len(dev)), index=dev.index)


def global_popularity_scores(dev: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Score each candidate by its total click count in train (no time window)."""
    click_counts = train.groupby("news_id")["label"].sum()
    return dev["news_id"].map(click_counts).fillna(0.0)


def _print_row(name: str, metrics: dict, elapsed: float) -> None:
    print(
        f"{name:<20}{metrics['auc']:>10.4f}{metrics['mrr']:>10.4f}"
        f"{metrics['ndcg@5']:>10.4f}{metrics['ndcg@10']:>10.4f}"
        f"{metrics['n_impressions']:>10}{metrics['n_skipped']:>10}"
        f"   ({elapsed:.1f}s)"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Print impression-averaged ranking metrics for random, "
        "global-popularity, recent-popularity, and logistic-regression "
        "scorers on the dev split."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=repo_root / "data" / "processed",
        help="Directory with train/dev samples and news parquet files "
        "(default: data/processed).",
    )
    args = parser.parse_args()

    train = pd.read_parquet(args.processed_dir / "train_samples.parquet")
    dev = pd.read_parquet(args.processed_dir / "dev_samples.parquet")
    news = pd.read_parquet(args.processed_dir / "news.parquet")

    header = (
        f"{'scorer':<20}{'auc':>10}{'mrr':>10}{'ndcg@5':>10}"
        f"{'ndcg@10':>10}{'n_impr':>10}{'n_skip':>10}"
    )
    print(header)

    def evaluate(name: str, scores) -> None:
        t0 = time.time()
        metrics = ranking_metrics(
            dev.assign(score=scores)[["impression_id", "label", "score"]]
        )
        _print_row(name, metrics, time.time() - t0)

    evaluate("random", random_scores(dev))
    evaluate("global_popularity", global_popularity_scores(dev, train))

    # Features are built once from a click index that only ever sees train,
    # and reused for every scorer below -- the same function, same index,
    # whether the candidates come from train (for fitting) or dev (for
    # scoring), which is the training/serving skew guarantee this project
    # is built around.
    click_index = build_click_time_index(train)
    print("building dev features...", file=sys.stderr)
    dev_features = build_features(dev, news, click_index)

    evaluate("recent_popularity_24h", dev_features["pop_24h"])

    print("building train features and fitting logistic regression...", file=sys.stderr)
    train_features = build_features(train, news, click_index)
    ranker = LogisticRegressionRanker().fit(train_features, train["label"])
    lr_scores = ranker.predict_proba(dev_features)
    evaluate("logistic_regression", lr_scores)


if __name__ == "__main__":
    main()
