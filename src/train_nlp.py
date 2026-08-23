"""Train and evaluate sentiment classifiers on TF-IDF features, save best model."""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from wordcloud import STOPWORDS, WordCloud

from utils import preprocess

logger = logging.getLogger(__name__)

LABELS = ["negative", "neutral", "positive"]


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


def ensure_outdir(path: str) -> None:
    """Create ``path`` (and parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    """Load a reviews CSV into a DataFrame."""
    return pd.read_csv(path)


def wordcloud_from_text(text: str, outpath: str) -> None:
    """Render a word cloud for ``text`` and save it as a PNG at ``outpath``."""
    wc = WordCloud(width=1000, height=600, background_color="white", stopwords=STOPWORDS)
    img = wc.generate(text).to_image()
    img.save(outpath, format="PNG")


def top_tfidf_features(
    vectorizer: TfidfVectorizer,
    clf: object,
    k: int = 25,
    outpath: str = "outputs/top_features.txt",
    labels: np.ndarray | None = None,
) -> None:
    """Write the top-``k`` TF-IDF features per class for ``clf`` to ``outpath``."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    lines = []
    if hasattr(clf, "coef_"):  # Logistic Regression (OvR)
        coefs = clf.coef_
        for idx, row in enumerate(coefs):
            top = np.argsort(row)[-k:][::-1]
            lab = labels[idx] if labels is not None else str(idx)
            lines.append(f"[{lab}] top +weights: " + ", ".join(feature_names[top]))
    elif hasattr(clf, "feature_importances_"):  # RandomForest
        importances = clf.feature_importances_
        top = np.argsort(importances)[-k:][::-1]
        lines.append("RandomForest top features: " + ", ".join(feature_names[top]))
    else:
        lines.append("Top features not available for this classifier.")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def validate_inputs(ap: argparse.ArgumentParser, df: pd.DataFrame, test_size: float) -> None:
    """Fail fast with a clear message instead of an opaque sklearn traceback."""
    if not 0.0 < test_size < 1.0:
        ap.error(f"--test-size must be between 0 and 1 (exclusive), got {test_size}")
    if df.empty:
        ap.error("No usable rows after dropping rows with missing text/label.")
    class_counts = df["label"].value_counts()
    if int(class_counts.min()) < 2:
        ap.error(
            "Each label needs at least 2 samples for a stratified train/test split; "
            f"found: {class_counts.to_dict()}"
        )
    n_test = round(len(df) * test_size)
    n_classes = df["label"].nunique()
    if n_test < n_classes:
        ap.error(
            f"--test-size {test_size} on {len(df)} rows yields {n_test} test row(s), "
            f"fewer than the {n_classes} classes present; increase --test-size or add data."
        )


def build_models(seed: int) -> dict[str, object]:
    """Instantiate the candidate classifiers to compare."""
    return {
        "nb": MultinomialNB(),
        "logreg": LogisticRegression(max_iter=3000),
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed),
    }


def evaluate_models(
    models: dict[str, object],
    tfidf: TfidfVectorizer,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, object]], str, Pipeline]:
    """Fit each model, score macro-F1 on the test split, return metrics + the best pipeline."""
    metrics: dict[str, dict[str, object]] = {}
    best_name: str | None = None
    best_score = -1.0
    best_pipe: Pipeline | None = None

    for name, clf in models.items():
        pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, digits=3, zero_division=0)
        metrics[name] = {"f1_macro": float(f1), "report": report}
        if f1 > best_score:
            best_score = f1
            best_name = name
            best_pipe = pipe

    assert best_pipe is not None and best_name is not None  # models is non-empty
    return metrics, best_name, best_pipe


def save_metrics_and_report(
    metrics: dict[str, dict[str, object]], best_name: str, outdir: str
) -> None:
    """Write metrics.json and classification_report.txt for the run."""
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"models": metrics, "best_model": best_name}, f, indent=2)
    with open(os.path.join(outdir, "classification_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(str(metrics[best_name]["report"]))


def save_confusion_matrix(
    best_pipe: Pipeline, X_test: pd.Series, y_test: pd.Series, outdir: str
) -> None:
    """Predict with the already-fitted best pipeline and save the confusion matrix plot."""
    y_pred_best = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (best model)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=160)
    plt.close(fig)


def save_wordclouds(df: pd.DataFrame, outdir: str) -> None:
    """Render positive/negative word clouds from the cleaned text."""
    text_pos = " ".join(df[df["label"] == "positive"]["text_clean"].tolist())
    text_neg = " ".join(df[df["label"] == "negative"]["text_clean"].tolist())
    wordcloud_from_text(text_pos, os.path.join(outdir, "wordcloud_positive.png"))
    wordcloud_from_text(text_neg, os.path.join(outdir, "wordcloud_negative.png"))


def save_model_artifacts(best_pipe: Pipeline, outdir: str) -> None:
    """Write top TF-IDF features and dump the fitted vectorizer + classifier."""
    best_vec = best_pipe.named_steps["tfidf"]
    best_clf = best_pipe.named_steps["clf"]
    top_tfidf_features(
        best_vec,
        best_clf,
        k=30,
        outpath=os.path.join(outdir, "top_features.txt"),
        labels=best_clf.classes_ if hasattr(best_clf, "classes_") else None,
    )
    dump(best_vec, os.path.join(outdir, "vectorizer.joblib"))
    dump(best_clf, os.path.join(outdir, "best_model.joblib"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default=os.environ.get("SENTIMENT_OUTDIR", "outputs"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=_env_int("SENTIMENT_SEED", 42))
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    df = load_dataset(args.input)
    df.dropna(subset=["text", "label"], inplace=True)
    validate_inputs(ap, df, args.test_size)
    df["text_clean"] = df["text"].astype(str).apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"],
        df["label"],
        test_size=args.test_size,
        stratify=df["label"],
        random_state=args.seed,
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    models = build_models(args.seed)
    metrics, best_name, best_pipe = evaluate_models(models, tfidf, X_train, y_train, X_test, y_test)

    save_metrics_and_report(metrics, best_name, args.outdir)
    save_confusion_matrix(best_pipe, X_test, y_test, args.outdir)
    save_wordclouds(df, args.outdir)
    save_model_artifacts(best_pipe, args.outdir)

    best_score = metrics[best_name]["f1_macro"]
    logger.info("[OK] Training complete.")
    logger.info("Best model: %s (F1-macro=%.3f)", best_name, best_score)
    logger.info("Outputs saved to: %s", args.outdir)


if __name__ == "__main__":
    main()
