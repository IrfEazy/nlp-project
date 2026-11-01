# NLP project

Classifies Yelp reviews from the Hugging Face `Yelp/yelp_review_full` dataset into 1-5 star classes (five classes; also 3-class, polarity and DBpedia transfer tasks). It is a PyTorch/scikit-learn port of the original notebook and covers Logistic Regression with Bag-of-Words or TF-IDF features, a character-level CNN, a BiLSTM with GloVe embeddings, DistilBERT fine-tuning and transfer, and zero-shot DeBERTa-v3 (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`).

## Results

5-class test accuracy, re-measured here versus the original notebook (values from [docs/research-report.md](docs/research-report.md)):

| Model | Re-measured | Notebook |
|---|---|---|
| LR BoW | 0.5771 | 0.57704 |
| LR TF-IDF | 0.5807 | 0.58068 |
| Character CNN | 0.6256 | 0.6228 |
| BiLSTM + GloVe | 0.6255 | 0.6132 (final table 0.6039) |
| DistilBERT fine-tune | 0.6855 | 0.684 |
| DistilBERT transfer | 0.5523 | 0.554 |
| Zero-shot DeBERTa | 0.4756 | 0.4755 |

Classical runs were on CPU, deep runs on Kaggle 2x T4, all single runs with seed 42. See the [research report](docs/research-report.md) for all 20 experiments and the deviations.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The first run downloads the dataset and NLTK stopwords, so it needs network access.

    uv sync                # add --extra deep for the PyTorch models
    uv run python -m src.main baseline 5000
    uv run pytest

Commands (the number is an optional sample size for faster runs):

    uv run python -m src.main baseline [sample_size]          # compare BoW and TF-IDF
    uv run python -m src.main single [model] [sample_size]    # one model (default bow; see AGENTS.md)
    uv run python -m src.main experiments                     # list named experiments
    uv run python -m src.main experiment <name> [sample_size] # run one
    uv run python -m src.main analyze [sample_size]           # dataset statistics
    uv run python -m src.main custom                          # hand-composed workflow demo

The deep experiments need the optional PyTorch extra (`uv sync --extra deep`) and download GloVe vectors or model checkpoints on first use. Full deep runs are practical only on a GPU; use small sample sizes on CPU. The `custom` mode is described in [AGENTS.md](AGENTS.md).

## More information

- [AGENTS.md](AGENTS.md): commands, layout, and project conventions (also read by coding agents).
- [src/AGENTS.md](src/AGENTS.md): architecture and how to add a model.
- [tests/AGENTS.md](tests/AGENTS.md): testing conventions.
- [docs/research-report.md](docs/research-report.md): notebook results versus re-measured results.
- [docs/known-issues.md](docs/known-issues.md): current known issues.

Released under the license in [LICENSE](LICENSE).
