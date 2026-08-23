import json
import os
import subprocess
import sys

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_fixture_csv(path: str) -> None:
    rows = []
    positive = ["great product love it", "amazing quality fast shipping", "excellent value"]
    negative = ["terrible broken item", "awful slow refund please", "bad quality waste"]
    neutral = ["okay average product", "fine does the job", "standard normal item"]
    for i, text in enumerate(positive * 4, start=1):
        rows.append([i, text, "positive"])
    offset = len(rows)
    for i, text in enumerate(negative * 4, start=offset + 1):
        rows.append([i, text, "negative"])
    offset = len(rows)
    for i, text in enumerate(neutral * 4, start=offset + 1):
        rows.append([i, text, "neutral"])
    pd.DataFrame(rows, columns=["review_id", "text", "label"]).to_csv(path, index=False)


@pytest.fixture
def fixture_csv(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    _make_fixture_csv(str(csv_path))
    return str(csv_path)


def test_train_nlp_end_to_end(fixture_csv, tmp_path):
    outdir = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(_ROOT, "src", "train_nlp.py"),
            "--input",
            fixture_csv,
            "--outdir",
            str(outdir),
            "--test-size",
            "0.3",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0 and "nltk" in (result.stderr or "").lower():
        pytest.skip(f"NLTK data unavailable in this environment: {result.stderr[-500:]}")
    assert result.returncode == 0, result.stderr

    for fname in (
        "metrics.json",
        "classification_report.txt",
        "confusion_matrix.png",
        "wordcloud_positive.png",
        "wordcloud_negative.png",
        "top_features.txt",
        "best_model.joblib",
        "vectorizer.joblib",
    ):
        assert (outdir / fname).exists(), f"missing output: {fname}"

    metrics = json.loads((outdir / "metrics.json").read_text())
    assert metrics["best_model"] in {"nb", "logreg", "rf"}
    assert set(metrics["models"].keys()) == {"nb", "logreg", "rf"}
    for model_metrics in metrics["models"].values():
        assert 0.0 <= model_metrics["f1_macro"] <= 1.0
