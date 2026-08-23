# Contributing

## Dev setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Before opening a PR
The CI workflow (`.github/workflows/ci.yml`) runs these checks; run them locally first:
```bash
ruff check src data
black --check src data
mypy src data
pytest tests
```

## Conventions
- Line length 100, enforced by `ruff` and `black` (`pyproject.toml`).
- Type-check with `mypy`; annotate new functions.
- Add tests for new behavior under `tests/`; `tests/conftest.py` puts `src/`
  and `data/` on the path so imports match how the scripts are run directly.
- `data/reviews.csv` and `outputs/` are generated, not committed — see the
  README's "Generate Synthetic Reviews" / "Train & Evaluate" sections.
- `--outdir` (train) and `--seed` / `--out` (generate) can be set via the
  `SENTIMENT_OUTDIR`, `SENTIMENT_SEED`, `SENTIMENT_DATA_OUT` environment
  variables; explicit CLI flags always take precedence.
