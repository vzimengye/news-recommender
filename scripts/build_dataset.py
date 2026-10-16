"""Parse raw MIND-small TSVs and write candidate-level Parquet samples."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Running this file directly (``python scripts/build_dataset.py``) only puts
# scripts/ on sys.path, not the repo root, so the src package can't be found
# without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_samples import build_samples  # noqa: E402
from src.data.parse import parse_behaviors, parse_news  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Parse MINDsmall_train/MINDsmall_dev behaviors.tsv and "
        "news.tsv, then write train/dev candidate samples and the "
        "deduplicated news table as Parquet."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Directory containing MINDsmall_train/ and MINDsmall_dev/ "
        "(default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "processed",
        help="Directory to write train_samples.parquet, dev_samples.parquet, "
        "and news.parquet (default: data/processed).",
    )
    args = parser.parse_args()

    train_dir = args.raw_dir / "MINDsmall_train"
    dev_dir = args.raw_dir / "MINDsmall_dev"
    required_files = {
        "train behaviors.tsv": train_dir / "behaviors.tsv",
        "train news.tsv": train_dir / "news.tsv",
        "dev behaviors.tsv": dev_dir / "behaviors.tsv",
        "dev news.tsv": dev_dir / "news.tsv",
    }
    missing = [label for label, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing raw MIND-small files: "
            + ", ".join(missing)
            + ". Download and extract MINDsmall_train.zip/MINDsmall_dev.zip "
            f"into {args.raw_dir} first (see README.md)."
        )

    t0 = time.time()
    train_behaviors = parse_behaviors(required_files["train behaviors.tsv"])
    dev_behaviors = parse_behaviors(required_files["dev behaviors.tsv"])
    train_news = parse_news(required_files["train news.tsv"])
    dev_news = parse_news(required_files["dev news.tsv"])
    print(f"parsed raw TSVs in {time.time() - t0:.1f}s")

    t0 = time.time()
    paths = build_samples(
        train_behaviors,
        dev_behaviors,
        train_news,
        dev_news,
        output_dir=args.output_dir,
    )
    print(f"built candidate samples in {time.time() - t0:.1f}s")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
