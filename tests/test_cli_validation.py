import os
import subprocess
import sys

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script_relpath, args):
    return subprocess.run(
        [sys.executable, os.path.join(_ROOT, *script_relpath.split("/")), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_generate_reviews_rejects_non_positive_n(tmp_path):
    result = _run(
        "data/generate_reviews.py",
        ["--n", "0", "--out", str(tmp_path / "out.csv")],
    )
    assert result.returncode != 0
    assert "--n must be a positive integer" in result.stderr
    assert not (tmp_path / "out.csv").exists()


def test_generate_reviews_rejects_negative_n(tmp_path):
    result = _run(
        "data/generate_reviews.py",
        ["--n", "-5", "--out", str(tmp_path / "out.csv")],
    )
    assert result.returncode != 0
    assert "--n must be a positive integer" in result.stderr


@pytest.fixture
def small_fixture_csv(tmp_path):
    df = pd.DataFrame(
        {
            "review_id": [1, 2, 3, 4],
            "text": ["great product", "bad product", "okay product", "fine item"],
            "label": ["positive", "negative", "neutral", "neutral"],
        }
    )
    path = tmp_path / "reviews.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_train_nlp_rejects_test_size_out_of_range(small_fixture_csv, tmp_path):
    result = _run(
        "src/train_nlp.py",
        [
            "--input",
            small_fixture_csv,
            "--outdir",
            str(tmp_path / "out"),
            "--test-size",
            "1.5",
        ],
    )
    assert result.returncode != 0
    assert "--test-size must be between 0 and 1" in result.stderr


def test_train_nlp_rejects_zero_test_size(small_fixture_csv, tmp_path):
    result = _run(
        "src/train_nlp.py",
        ["--input", small_fixture_csv, "--outdir", str(tmp_path / "out"), "--test-size", "0"],
    )
    assert result.returncode != 0
    assert "--test-size must be between 0 and 1" in result.stderr


def test_train_nlp_rejects_single_member_class(tmp_path):
    df = pd.DataFrame(
        {
            "review_id": [1, 2, 3],
            "text": ["great product", "bad product", "okay product"],
            "label": ["positive", "negative", "neutral"],  # each class has only 1 sample
        }
    )
    csv_path = tmp_path / "reviews.csv"
    df.to_csv(csv_path, index=False)
    result = _run(
        "src/train_nlp.py",
        ["--input", str(csv_path), "--outdir", str(tmp_path / "out"), "--test-size", "0.2"],
    )
    assert result.returncode != 0
    assert "at least 2 samples" in result.stderr
