import os
import subprocess
import sys

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script_relpath, args, env=None):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, os.path.join(_ROOT, *script_relpath.split("/")), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=full_env,
    )


def test_generate_reviews_env_seed_matches_explicit_flag(tmp_path):
    out_env = tmp_path / "env.csv"
    out_flag = tmp_path / "flag.csv"
    _run(
        "data/generate_reviews.py",
        ["--n", "30", "--out", str(out_env)],
        env={"SENTIMENT_SEED": "99"},
    )
    _run("data/generate_reviews.py", ["--n", "30", "--seed", "99", "--out", str(out_flag)])
    assert out_env.read_text() == out_flag.read_text()


def test_generate_reviews_explicit_flag_overrides_env(tmp_path):
    out_env_flag = tmp_path / "override.csv"
    out_flag_only = tmp_path / "flag_only.csv"
    _run(
        "data/generate_reviews.py",
        ["--n", "30", "--seed", "5", "--out", str(out_env_flag)],
        env={"SENTIMENT_SEED": "99"},
    )
    _run("data/generate_reviews.py", ["--n", "30", "--seed", "5", "--out", str(out_flag_only)])
    assert out_env_flag.read_text() == out_flag_only.read_text()


def test_generate_reviews_invalid_env_seed_falls_back(tmp_path):
    out = tmp_path / "fallback.csv"
    result = _run(
        "data/generate_reviews.py",
        ["--n", "10", "--out", str(out)],
        env={"SENTIMENT_SEED": "not-a-number"},
    )
    assert result.returncode == 0
    assert "Ignoring invalid SENTIMENT_SEED" in result.stderr
    assert out.exists()


@pytest.fixture
def small_fixture_csv(tmp_path):
    rows = []
    labels = ["positive", "negative", "neutral"]
    texts = {
        "positive": ["great product", "amazing quality", "love it", "excellent buy"],
        "negative": ["bad product", "terrible quality", "awful", "broken item"],
        "neutral": ["okay product", "fine item", "average", "standard"],
    }
    review_id = 1
    for label in labels:
        for text in texts[label]:
            rows.append([review_id, text, label])
            review_id += 1
    path = tmp_path / "reviews.csv"
    pd.DataFrame(rows, columns=["review_id", "text", "label"]).to_csv(path, index=False)
    return str(path)


def test_train_nlp_env_outdir_is_used(small_fixture_csv, tmp_path):
    outdir = tmp_path / "env_outdir"
    result = _run(
        "src/train_nlp.py",
        ["--input", small_fixture_csv, "--test-size", "0.3", "--seed", "1"],
        env={"SENTIMENT_OUTDIR": str(outdir)},
    )
    if result.returncode != 0 and "nltk" in (result.stderr or "").lower():
        pytest.skip(f"NLTK data unavailable in this environment: {result.stderr[-500:]}")
    assert result.returncode == 0, result.stderr
    assert (outdir / "metrics.json").exists()
