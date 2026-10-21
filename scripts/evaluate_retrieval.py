"""Print recall@K for retrievers on a sample of dev clicks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.retrieval.popular import PopularityRetriever  # noqa: E402

RANDOM_SEED = 0
K_VALUES = (50, 100, 200)
DEFAULT_SAMPLE_SIZE = 1500


def random_retriever(news: pd.DataFrame) -> callable:
    """A retriever that returns k random available news IDs -- a sanity
    check for recall_at_k itself: it should score far below the real
    retriever, or the evaluation code is suspect, not the model."""
    news_ids = news["news_id"].to_numpy()
    first_seen = pd.to_datetime(news["first_seen"]).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)

    def retrieve(user_id: str, time_: object, k: int) -> list[str]:
        available = news_ids[first_seen <= np.datetime64(pd.Timestamp(time_))]
        if len(available) == 0:
            return []
        k = min(k, len(available))
        return list(rng.choice(available, size=k, replace=False))

    return retrieve


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Print recall@K for retrievers on a sample of dev clicks."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=repo_root / "data" / "processed"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of dev click events to sample for recall@K "
        f"(default: {DEFAULT_SAMPLE_SIZE}). The Python-level retrieval loop "
        "is too slow to run on all ~111k dev clicks in a reasonable time.",
    )
    args = parser.parse_args()

    train = pd.read_parquet(
        args.processed_dir / "train_samples.parquet",
        columns=["news_id", "time", "label"],
    )
    news = pd.read_parquet(
        args.processed_dir / "news.parquet", columns=["news_id", "first_seen"]
    )
    dev = pd.read_parquet(
        args.processed_dir / "dev_samples.parquet",
        columns=["user_id", "time", "news_id", "label"],
    )

    clicks = dev.loc[dev["label"] == 1]
    sample = clicks.sample(
        n=min(args.sample_size, len(clicks)), random_state=RANDOM_SEED
    )
    queries = list(zip(sample["user_id"], sample["time"], sample["news_id"]))
    print(f"evaluating on {len(queries)} sampled dev clicks (of {len(clicks)} total)")

    popularity = PopularityRetriever(window_hours=24).fit(train, news)
    retrievers = {
        "random": random_retriever(news),
        "popularity_24h": popularity.retrieve,
    }

    max_k = max(K_VALUES)
    header = f"{'retriever':<20}" + "".join(f"{'recall@' + str(k):>12}" for k in K_VALUES)
    print(header)
    for name, retrieve_fn in retrievers.items():
        t0 = time.time()
        # Retrieve once at the largest k per query; recall@50 and recall@100
        # are then just hits within a prefix of that same ranked list, so
        # this is one retrieve() call per query, not one per (query, k).
        hits = {k: 0 for k in K_VALUES}
        for user_id, query_time, true_news_id in queries:
            retrieved = retrieve_fn(user_id, query_time, max_k)
            for k in K_VALUES:
                if true_news_id in retrieved[:k]:
                    hits[k] += 1
        elapsed = time.time() - t0
        recalls = [hits[k] / len(queries) for k in K_VALUES]
        print(
            f"{name:<20}" + "".join(f"{value:>12.4f}" for value in recalls)
            + f"   ({elapsed:.1f}s)"
        )


if __name__ == "__main__":
    main()
