"""Checks against the real downloaded MIND-small archives.

Unlike test_parse.py/test_build_samples.py/test_split.py, which verify our
own code against small fixtures, this file verifies an assumption about the
*official* dataset itself: that the dev split is entirely after the train
split in time, so using dev as a held-out evaluation set is not leaking
future information into training. These tests are skipped when the raw
archives have not been downloaded (data/raw/ is gitignored and not present
on a fresh clone).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.parse import parse_behaviors

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TRAIN_BEHAVIORS = RAW_DIR / "MINDsmall_train" / "behaviors.tsv"
DEV_BEHAVIORS = RAW_DIR / "MINDsmall_dev" / "behaviors.tsv"

requires_mind_data = pytest.mark.skipif(
    not (TRAIN_BEHAVIORS.is_file() and DEV_BEHAVIORS.is_file()),
    reason="requires the downloaded MINDsmall_train/dev archives under data/raw/",
)


@requires_mind_data
def test_official_dev_split_starts_after_train_ends() -> None:
    train = parse_behaviors(TRAIN_BEHAVIORS)
    dev = parse_behaviors(DEV_BEHAVIORS)

    assert dev["time"].min() > train["time"].max()
