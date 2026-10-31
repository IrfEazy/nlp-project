# NLP project

Classifies Yelp reviews into 5 star-rating classes (labels 0-4) using the Hugging Face `yelp_review_full` dataset, with scikit-learn Bag-of-Words or TF-IDF features and Logistic Regression. The code is a small layered package under `src/` driven by a step-based pipeline.

## Stack and commands

Python 3.12, managed with `uv`. Run everything from the repository root.

- `uv sync` installs dependencies.
- `uv run python -m src.main baseline [sample_size]` compares BoW and TF-IDF (default mode).
- `uv run python -m src.main single [bow|tfidf]` runs one model with plots. Model and sample size cannot be combined; see [known issues](docs/known-issues.md#behavior).
- `uv run python -m src.main custom` runs the hand-composed workflow demo.
- `uv run pytest` runs all tests; `uv run pytest -m unit` runs only unit tests; `uv run pytest --cov=src` adds coverage.

The first real run needs network access: it downloads the HF dataset and the NLTK stopwords corpus.

## Layout

- `src/config.py`: configuration dataclasses and experiment factories.
- `src/main.py`: command-line entry point.
- `src/data/`: dataset loading and statistics.
- `src/preprocessing/`: text cleaning and vectorizer factory.
- `src/models/`: model strategies and the model factory.
- `src/pipeline/`: pipeline steps and workflow builder.
- `src/evaluation/`: metrics and plots.
- `src/utils/`: reproducibility setup.
- `tests/`: pytest suite mirroring `src/`.

## Where to look

- `src/AGENTS.md`: module responsibilities, dependency direction, context contract, how to add a model.
- `tests/AGENTS.md`: test conventions, markers, fixtures, mocking rules.
- `docs/known-issues.md`: current known oddities, not yet fixed.

## Do not read whole

- `yelp-reviews-project-original.py`: legacy notebook export of about 144 KB, imported nowhere, reference only. Grep it instead.
- `uv.lock`: large generated lockfile. Grep it instead.

## Instruction files

- Keep each fact in exactly one file; put component details in that directory's `AGENTS.md` and link to them from here.
- Do not add `CLAUDE.md`, `CLAUDE.local.md`, `.claude/CLAUDE.md` or `.claude/rules` here; they would suppress native `AGENTS.md` loading.
- Keep each instruction file under 200 lines.
- Update the docs in the same change as the code.
