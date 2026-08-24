<div align="center">
           
# Sentiment Analysis (NLP)

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20NB%2FLogReg%2FRF-orange)
![Testing](https://img.shields.io/badge/Tests-pytest-green)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Sentiment-Analysis-NLP/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Sentiment-Analysis-NLP/actions/workflows/ci.yml)

</div>

A customer-review sentiment classifier built on a **TF-IDF + classical ML** pipeline (Naive Bayes, Logistic Regression, Random Forest), with **synthetic data generation**, **macro-F1 model selection**, **visual reports**, **environment-driven configuration**, and an **automated quality gate** (ruff, black, mypy, pytest, CI).

> **Important:** This project trains on a **synthetic, templated review dataset**, not real customer reviews.
>
> The generator samples from a small set of seed sentences, so the classes are close to linearly separable and models routinely reach macro-F1 = 1.000 on the holdout set. That number demonstrates a correct, honest ML workflow, it is not evidence of real-world sentiment-classification accuracy. See [Limitations](#limitations).

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Model Output](#model-output)
- [Model Artifacts and Loading Safety](#model-artifacts-and-loading-safety)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Sentiment classification is a standard first NLP project, but it's easy to build one that looks good on paper without being honest about what the numbers mean. This project is a small, complete reference implementation that:

- generates a labeled dataset instead of requiring one,
- preprocesses and vectorizes text with a documented, testable pipeline,
- compares multiple classifiers on a common holdout split and picks the best by macro-F1,
- produces visual and text reports for inspection, not just a single score, and
- is honest in its own documentation about why that score is close to perfect and what that does and doesn't prove.

The goal is a clean, fully-tested TF-IDF classification workflow that's easy to read end to end, not a benchmark-chasing model.

---

## What This Project Does

This project can:

- Generate a reproducible synthetic review dataset (`--seed` is fully deterministic)
- Clean and lemmatize review text (lowercasing, URL/punctuation stripping, stopword removal, lemmatization)
- Vectorize text with TF-IDF (unigrams + bigrams)
- Train and compare Multinomial Naive Bayes, Logistic Regression, and Random Forest
- Select the best model by macro-F1 on a stratified holdout split
- Save a confusion matrix, positive/negative word clouds, and top discriminative features
- Save the fitted vectorizer and classifier as checked, versioned artifacts
- Validate CLI inputs up front with clear error messages instead of raw stack traces
- Read key defaults (seed, output paths) from environment variables
- Run a full lint/type/test gate locally and in CI

---

## What This Project Does Not Do

This project does **not**:

- Train on real customer reviews, the data is synthetically generated from a small seed set
- Generalize claims to production sentiment analysis without new, real-world data
- Handle multiple languages, sarcasm, or domain-specific jargon
- Serve predictions via an API or web app, it's a CLI training/reporting pipeline
- Guarantee the near-perfect holdout scores hold on harder, real data

---

## Key Features

- **TF-IDF vectorization** with unigrams + bigrams (`min_df=2`, up to 50,000 features)
- **Three-model comparison**: Multinomial Naive Bayes, Logistic Regression, Random Forest
- **Macro-F1 model selection** on a stratified train/test split
- **Deterministic data generation**, fixed `--seed` reproduces byte-identical output
- **Fail-fast CLI validation** (bad `--test-size`, empty data, single-member classes)
- **Environment-driven config** (`SENTIMENT_SEED`, `SENTIMENT_OUTDIR`, `SENTIMENT_DATA_OUT`)
- **Visual + text reports**: confusion matrix, word clouds, top TF-IDF features per class
- **Documented artifact-loading risk**, `joblib.load` trust boundary called out explicitly
- **24 automated tests** covering preprocessing, generation, CLI validation, env config, and an end-to-end training smoke test
- **CI-enforced quality gate**: ruff, black, mypy, pytest on every push/PR

---

## System Workflow

```text
Raw customer review text
        ↓
Text preprocessing (utils.preprocess: clean, tokenize, stopword removal, lemmatize)
        ↓
TF-IDF vectorization (unigrams + bigrams)
        ↓
Naive Bayes / Logistic Regression / Random Forest (trained in parallel)
        ↓
Macro-F1 comparison → best model selected
        ↓
Confusion matrix, word clouds, top features
        ↓
Saved artifacts (best_model.joblib, vectorizer.joblib, metrics.json)
```

---

## Project Structure

```text
Sentiment-Analysis-NLP/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── generate_reviews.py
│   └── reviews.csv            (generated, gitignored)
│
├── outputs/                   (generated, gitignored)
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   ├── top_features.txt
│   ├── best_model.joblib
│   └── vectorizer.joblib
│
├── src/
│   ├── train_nlp.py
│   └── utils.py
│
├── tests/
│   ├── conftest.py
│   ├── test_cli_validation.py
│   ├── test_env_config.py
│   ├── test_generate_reviews.py
│   ├── test_preprocess.py
│   ├── test_train_nlp_helpers.py
│   └── test_train_smoke.py
│
├── .gitignore
├── .python-version
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Sentiment-Analysis-NLP.git
cd Sentiment-Analysis-NLP
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development tools (ruff, black, mypy, pytest):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

Generate the synthetic dataset:

```bash
python data/generate_reviews.py --n 8000 --seed 42 --out data/reviews.csv
```

Train and evaluate:

```bash
python src/train_nlp.py --input data/reviews.csv --outdir outputs --test-size 0.2 --seed 42
```

---

## Configuration

CLI flags always take precedence; these environment variables set the defaults when a flag is omitted:

<div align="center">

| Variable | Used by | Default |
|---|---|---|
| `SENTIMENT_SEED` | both scripts (`--seed`) | `42` |
| `SENTIMENT_OUTDIR` | `train_nlp.py` (`--outdir`) | `outputs` |
| `SENTIMENT_DATA_OUT` | `generate_reviews.py` (`--out`) | `data/reviews.csv` |

</div>

An invalid `SENTIMENT_SEED` (non-integer) is ignored with a logged warning, falling back to `42`, rather than crashing.

---

## Model Output

`train_nlp.py` writes `outputs/metrics.json` with per-model macro-F1 and the selected best model:

```json
{
  "models": {
    "nb":     { "f1_macro": 1.0, "report": "..." },
    "logreg": { "f1_macro": 1.0, "report": "..." },
    "rf":     { "f1_macro": 1.0, "report": "..." }
  },
  "best_model": "nb"
}
```

**Other generated outputs:**

- `classification_report.txt` | precision/recall/F1 per class for the best model
- `confusion_matrix.png` | best model's predictions vs. true labels on the holdout set
- `wordcloud_positive.png`, `wordcloud_negative.png` | most frequent cleaned terms per class
- `top_features.txt` | top TF-IDF features per class (Logistic Regression / Random Forest only | Naive Bayes has neither `coef_` nor `feature_importances_`, so this file will read "Top features not available for this classifier" when NB is the selected model, which it usually is on this dataset)
- `best_model.joblib`, `vectorizer.joblib` | the fitted classifier and vectorizer

---

## Model Artifacts and Loading Safety

`best_model.joblib` and `vectorizer.joblib` are loaded with `joblib.load`, which like `pickle`, **can execute arbitrary code** for a maliciously crafted file. Only load `.joblib` artifacts you generated yourself or that come from a source you fully trust; never one downloaded from an untrusted source.

---

## Evaluation Metrics

Evaluation uses a stratified train/test split (`--test-size`, default 0.2) with the same split shared across all three models for a fair comparison.

<div align="center">

| Metric | Why it matters |
|---|---|
| Accuracy | Overall correctness at the default decision rule |
| Macro F1 | Balance across negative / neutral / positive, unweighted by class size |
| Precision / Recall (per class) | Per-class prediction quality, hidden by accuracy alone |

</div>

Example results from a full run (`--n 8000 --seed 42 --test-size 0.2`):

<div align="center">

| Model | Macro F1 (holdout) |
|---|---:|
| Multinomial Naive Bayes | 1.000 |
| Logistic Regression | 1.000 |
| Random Forest | 1.000 |

</div>

> All three models reach macro-F1 = 1.000 on this dataset. That's expected, not a sign of a strong real-world model, see [Limitations](#limitations) for why.

---

## Visual Reports

### Confusion Matrix

<div align="center">
<img width="600" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/feac74b0-cdf1-467d-add0-97535d9ab8b9" />
</div>

### Word Clouds

<div align="center">

| Positive Reviews | Negative Reviews |
|---|---|
| <img width="400" alt="wordcloud_positive" src="https://github.com/user-attachments/assets/698eeb83-3975-46b9-8173-3bb5228be4cb" /> | <img width="400" alt="wordcloud_negative" src="https://github.com/user-attachments/assets/081da50c-c54d-4ab2-915e-b61799a821ba" /> |

</div>

---

## Testing and CI

Run the full local gate:

```bash
ruff check src data
black --check src data
mypy src data
pytest tests
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs the same four checks on every push and pull request to `main`.

24 tests cover:

- text preprocessing (`test_preprocess.py`)
- synthetic data generation and determinism (`test_generate_reviews.py`)
- CLI input validation (`test_cli_validation.py`)
- environment-variable config and override precedence (`test_env_config.py`)
- `wordcloud_from_text` and both branches of `top_tfidf_features` (`test_train_nlp_helpers.py`)
- an end-to-end training run on a tiny fixture (`test_train_smoke.py`)

---

## Code Quality

The project separates responsibilities across a small number of focused modules and functions:

<div align="center">

| Module / function | Purpose |
|---|---|
| `src/utils.py` | Text cleaning, stopword removal, lemmatization (`preprocess`) |
| `src/train_nlp.py::build_models` | Instantiates the three candidate classifiers |
| `src/train_nlp.py::evaluate_models` | Fits each model, scores macro-F1, returns the best pipeline |
| `src/train_nlp.py::save_metrics_and_report` | Writes `metrics.json` and the classification report |
| `src/train_nlp.py::save_confusion_matrix` | Renders the confusion matrix from the already-fitted best model |
| `src/train_nlp.py::save_wordclouds` | Renders the positive/negative word clouds |
| `src/train_nlp.py::save_model_artifacts` | Writes top features and dumps the vectorizer/classifier |
| `data/generate_reviews.py::synthesize` | Deterministic synthetic dataset generation |

</div>

Tooling is configured through `pyproject.toml` (ruff, black, mypy, pytest) and `requirements-dev.txt`.

---

## Limitations

This project has important limitations:

- The dataset is synthetically generated from ~15 seed sentences per class plus a small pool of extra words, it is not real customer review text
- Because the seed sentences are close to linearly separable after TF-IDF, all three models reach macro-F1 = 1.000 on the holdout set; this reflects the dataset's simplicity, not model quality
- `top_features.txt` reports "not available" whenever the selected best model is Naive Bayes (no `coef_`/`feature_importances_`), which is the common case here
- No real-world validation, calibration, or fairness review has been done
- The project does not handle multiple languages, sarcasm, negation edge cases, or domain-specific vocabulary beyond what preprocessing already strips

The project is strongest as a clean, fully-tested reference implementation of a TF-IDF classification workflow, not as a production sentiment model.

---

## Responsible Use

This repository is intended for:

- machine learning and NLP education
- practicing TF-IDF text-classification workflows
- demonstrating a tested, CI-enforced ML project structure
- portfolio demonstration

It should not be used as-is for:

- classifying real customer reviews in production without retraining on real, representative data
- any decision with legal, financial, or safety consequences
- automated content moderation

---

## Future Improvements

Potential next improvements:

- Add a real, labeled review dataset option alongside the synthetic generator
- Add cross-validation instead of a single holdout split
- Add calibration metrics (Brier score, reliability plots)
- Add a small CLI or Streamlit demo for interactive predictions
- Add checksum-sidecar verification for saved artifacts (mirroring the trust-boundary note above with an enforced check)
- Explore transformer-based embeddings as a stronger baseline

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- wordcloud
- joblib
- pytest
- ruff, black, mypy
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

MIT | see [LICENSE](LICENSE).
