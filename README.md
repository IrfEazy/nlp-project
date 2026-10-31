# NLP project

Predicts the 1-5 star rating of Yelp reviews (five classes) from the Hugging Face `yelp_review_full` dataset, comparing Bag-of-Words and TF-IDF features with scikit-learn Logistic Regression.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The first run downloads the dataset and NLTK stopwords, so it needs network access.

    uv sync
    uv run python -m src.main baseline 5000
    uv run pytest

The `baseline` argument compares both models; the number is an optional sample size for faster runs. Other modes (`single`, `custom`) are described in [AGENTS.md](AGENTS.md).

## More information

- [AGENTS.md](AGENTS.md): commands, layout, and project conventions (also read by coding agents).
- [src/AGENTS.md](src/AGENTS.md): architecture and how to add a model.
- [tests/AGENTS.md](tests/AGENTS.md): testing conventions.
- [docs/known-issues.md](docs/known-issues.md): current known issues.

Released under the license in [LICENSE](LICENSE).
