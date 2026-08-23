import pytest

try:
    from utils import preprocess

    preprocess("warm-up")  # triggers NLTK corpus download; skip cleanly if unavailable
except Exception as exc:  # noqa: BLE001 - environment/network dependent
    pytest.skip(f"NLTK data unavailable: {exc}", allow_module_level=True)


def test_lowercases_and_strips_punctuation():
    assert preprocess("AMAZING!!! Product.") == "amazing product"


def test_removes_urls():
    result = preprocess("Check this out http://example.com great deal")
    assert "http" not in result
    assert "example" not in result


def test_removes_short_words_and_stopwords():
    result = preprocess("this is a great product")
    assert "is" not in result.split()
    assert "great" in result
    assert "product" in result


def test_lemmatizes_words():
    result = preprocess("The shoes were amazing")
    assert "shoe" in result.split()


def test_empty_string_returns_empty():
    assert preprocess("") == ""
