"""Generate a synthetic customer-review dataset for sentiment analysis."""

import argparse
import logging
import os
import random

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POSITIVE_SEEDS = [
    "Amazing quality and fast delivery! Totally satisfied.",
    "Great value for the price. Will buy again.",
    "Customer support was helpful and polite.",
    "I love this product. Highly recommended.",
    "Exceeded my expectations in every way.",
]
NEGATIVE_SEEDS = [
    "Terrible experience. The item arrived broken.",
    "Very poor quality and slow shipping.",
    "Not worth the money. I want a refund.",
    "Support was unhelpful and rude.",
    "Completely disappointed. Do not recommend.",
]
NEUTRAL_SEEDS = [
    "Works as expected. Nothing special.",
    "Okay product, decent for everyday use.",
    "Average experience overall.",
    "The item matches the description.",
    "It's fine, does the job.",
]


def _env_int(name: str, default: int) -> int:
    """Read an int default from the environment, falling back if unset/invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default %s", name, raw, default)
        return default


def synthesize(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate ``n`` synthetic labeled reviews, deterministic for a given ``seed``."""
    rng = np.random.default_rng(seed)
    py_random = random.Random(seed)
    labels = rng.choice(["positive", "neutral", "negative"], size=n, p=[0.45, 0.25, 0.30])
    rows = []
    for i, lab in enumerate(labels, start=1):
        if lab == "positive":
            base = py_random.choice(POSITIVE_SEEDS)
            extras = ["fast", "reliable", "excellent", "love", "five stars", "great"]
        elif lab == "negative":
            base = py_random.choice(NEGATIVE_SEEDS)
            extras = ["late", "broken", "bad", "refund", "waste", "one star"]
        else:
            base = py_random.choice(NEUTRAL_SEEDS)
            extras = ["okay", "average", "standard", "fine", "works", "normal"]
        words = base.split()
        words += rng.choice(extras, size=rng.integers(0, 6), replace=True).tolist()
        if rng.random() < 0.1:
            words.append("http://example.com")
        if rng.random() < 0.1:
            words.append("👍")
        if rng.random() < 0.1:
            words.append("#deal")
        text = " ".join(words)
        rows.append([i, text, lab])
    return pd.DataFrame(rows, columns=["review_id", "text", "label"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=_env_int("SENTIMENT_SEED", 42))
    ap.add_argument(
        "--out", type=str, default=os.environ.get("SENTIMENT_DATA_OUT", "data/reviews.csv")
    )
    args = ap.parse_args()

    if args.n < 1:
        ap.error(f"--n must be a positive integer, got {args.n}")

    df = synthesize(args.n, args.seed)
    df.to_csv(args.out, index=False)
    logger.info("[OK] wrote %s with %s rows", args.out, f"{len(df):,}")


if __name__ == "__main__":
    main()
