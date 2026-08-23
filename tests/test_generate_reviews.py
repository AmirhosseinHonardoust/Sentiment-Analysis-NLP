from generate_reviews import NEGATIVE_SEEDS, NEUTRAL_SEEDS, POSITIVE_SEEDS, synthesize


def test_same_seed_is_fully_reproducible():
    df1 = synthesize(200, seed=42)
    df2 = synthesize(200, seed=42)
    assert df1.equals(df2)


def test_different_seed_differs():
    df1 = synthesize(200, seed=42)
    df2 = synthesize(200, seed=7)
    assert not df1["text"].equals(df2["text"])


def test_shape_and_columns():
    df = synthesize(50, seed=1)
    assert list(df.columns) == ["review_id", "text", "label"]
    assert len(df) == 50
    assert set(df["label"].unique()) <= {"positive", "neutral", "negative"}


def test_review_ids_are_sequential():
    df = synthesize(30, seed=1)
    assert list(df["review_id"]) == list(range(1, 31))


def test_text_uses_expected_seed_sentences():
    df = synthesize(100, seed=1)
    all_seeds = POSITIVE_SEEDS + NEGATIVE_SEEDS + NEUTRAL_SEEDS
    for text in df["text"]:
        assert any(text.startswith(seed.split()[0]) for seed in all_seeds)
