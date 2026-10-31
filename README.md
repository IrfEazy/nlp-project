# NLP project

Predicts the 1-5 star rating of Yelp reviews (five classes) from the Hugging Face `Yelp/yelp_review_full` dataset, comparing Bag-of-Words and TF-IDF features with scikit-learn Logistic Regression.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The first run downloads the dataset and NLTK stopwords, so it needs network access.

    uv sync                # add --extra deep for the PyTorch models
    uv run python -m src.main baseline 5000
    uv run pytest

The `baseline` argument compares both models; the number is an optional sample size for faster runs. `single [model] [sample_size]` runs one model (`bow` or `tfidf`). `experiments` lists named experiments, `experiment <name> [sample_size]` runs one and `analyze [sample_size]` prints dataset statistics. The deep experiments (`cnn-char`, `cnn-char-mixed-case`, `lstm-glove`, `lstm-glove-3class`; the LSTM downloads the 376 MB GloVe vectors on first use) need the optional PyTorch extra: `uv sync --extra deep`. The `custom` mode is described in [AGENTS.md](AGENTS.md).

## More information

- [AGENTS.md](AGENTS.md): commands, layout, and project conventions (also read by coding agents).
- [src/AGENTS.md](src/AGENTS.md): architecture and how to add a model.
- [tests/AGENTS.md](tests/AGENTS.md): testing conventions.
- [docs/known-issues.md](docs/known-issues.md): current known issues.

Released under the license in [LICENSE](LICENSE).
