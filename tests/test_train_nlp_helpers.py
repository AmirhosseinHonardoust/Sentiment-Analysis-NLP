import numpy as np

from train_nlp import top_tfidf_features, wordcloud_from_text


def test_wordcloud_from_text_creates_valid_png(tmp_path):
    outpath = tmp_path / "cloud.png"
    wordcloud_from_text("great product amazing quality fast shipping love it", str(outpath))
    assert outpath.exists()
    with open(outpath, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


class _FakeVectorizer:
    def __init__(self, feature_names):
        self._feature_names = np.array(feature_names)

    def get_feature_names_out(self):
        return self._feature_names


class _FakeLogRegClf:
    def __init__(self, coef):
        self.coef_ = np.array(coef)


class _FakeRandomForestClf:
    def __init__(self, importances):
        self.feature_importances_ = np.array(importances)


class _FakeUnknownClf:
    pass


def test_top_tfidf_features_logreg_branch(tmp_path):
    vec = _FakeVectorizer(["bad", "good", "okay"])
    clf = _FakeLogRegClf([[0.1, 0.9, 0.2], [0.8, 0.1, 0.3]])
    outpath = tmp_path / "top.txt"
    top_tfidf_features(vec, clf, k=2, outpath=str(outpath), labels=["positive", "negative"])
    content = outpath.read_text()
    assert "[positive] top +weights:" in content
    assert "[negative] top +weights:" in content
    assert "good" in content  # highest weight for the positive row


def test_top_tfidf_features_random_forest_branch(tmp_path):
    vec = _FakeVectorizer(["bad", "good", "okay"])
    clf = _FakeRandomForestClf([0.1, 0.7, 0.2])
    outpath = tmp_path / "top.txt"
    top_tfidf_features(vec, clf, k=2, outpath=str(outpath))
    content = outpath.read_text()
    assert "RandomForest top features:" in content
    assert "good" in content  # highest importance


def test_top_tfidf_features_unknown_classifier(tmp_path):
    vec = _FakeVectorizer(["bad", "good", "okay"])
    clf = _FakeUnknownClf()
    outpath = tmp_path / "top.txt"
    top_tfidf_features(vec, clf, k=2, outpath=str(outpath))
    assert "not available" in outpath.read_text()
