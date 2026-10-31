# NLP project

Classifies Yelp reviews into 5 star-rating classes (labels 0-4) using the Hugging Face `Yelp/yelp_review_full` dataset, with scikit-learn Bag-of-Words or TF-IDF features and Logistic Regression. The code is a small layered package under `src/` driven by a step-based pipeline.

## Stack and commands

Python 3.12, managed with `uv`. Run everything from the repository root.

- `uv sync` installs dependencies; `uv sync --extra deep` adds the optional PyTorch extra (a plain `uv sync` removes it, and the deep tests then skip). Experiments `cnn-char`, `cnn-char-mixed-case`, `lstm-glove` and `lstm-glove-3class` need it; a missing extra exits 1 naming this command.
- `uv run python -m src.main baseline [sample_size]` compares BoW and TF-IDF (default mode).
- `uv run python -m src.main single [model] [sample_size]` runs one model with plots (model defaults to `bow`; `single 1000` is valid); an unknown model, invalid (non-integer or non-positive) sample size or extra argument exits 1.
- `uv run python -m src.main custom` runs the hand-composed workflow demo.
- `uv run python -m src.main experiments` lists the named experiments (`lr-bow`, `lr-tfidf`, `lr-bow-3class`, `lr-bow-polarity`, `lr-bow-no-neutral`, `lr-bow-dbpedia`); `experiment <name> [sample_size]` runs one (summary, classification report, top-10 features per class, plots); `analyze [sample_size]` prints dataset statistics (downloads the NLTK stopwords corpus). Unknown names, bad sample sizes and extra arguments exit 1.
- Smoke run: `MPLBACKEND=Agg uv run python -m src.main single bow 1000` (Agg avoids blocking plot windows when headless). It downloads the dataset. Observed 2026-10-31: the unauthenticated HF warning without `HF_TOKEN` is harmless, and accuracy near 0.33 on 1000 samples is expected and is not a quality measure.
- `uv run pytest` runs all tests; `uv run pytest -m unit` runs only unit tests; `uv run pytest --cov=src` adds coverage.

The first real run needs network access: it downloads the HF dataset and the NLTK stopwords corpus; the LSTM experiments also fetch the NLTK punkt data and the ~376 MB GloVe vectors (gensim cache).

## Dependencies

- Manage them only with `uv add`, `uv add --dev` and `uv remove`; never hand-edit dependency lists or `uv.lock`.
- Add packages by bare name so uv writes its own `>=` lower bound: no exact pins, no `--raw` or `--bounds`.
- Runtime dependencies are exactly what `src/` imports directly: declare every direct import (deep-learning ones such as `torch` go in the optional `deep` extra via `uv add --optional deep <pkg>`, imported only inside `src/models/deep/`; except the guarded optional torch/tensorflow imports in `src/utils/reproducibility.py`, see docs/known-issues.md), remove unused ones. Test tools go in the `dev` group.
- `uv.lock` is tracked; commit it with `pyproject.toml`.
- After a dependency change run the tests and the smoke run: re-resolution can bring major upgrades (see [src/AGENTS.md rules](src/AGENTS.md#rules)).

## Working style

Docs, code and commit messages in English; reply to the user in the language they write in (currently Italian).

## Layout

- `src/config.py`: configuration dataclasses and experiment factories.
- `src/main.py`: command-line entry point.
- `src/data/`: dataset loading, label tasks and statistics.
- `src/preprocessing/`: text cleaning and vectorizer factory.
- `src/models/`: model strategies and the model factory; `src/models/deep/`: PyTorch models and the shared training loop (`deep` extra).
- `src/pipeline/`: pipeline steps, workflow builder and experiment/analysis runners.
- `src/evaluation/`: metrics and plots.
- `src/utils/`: reproducibility setup.
- `tests/`: pytest suite mirroring `src/`.

## Where to look

- `src/AGENTS.md`: module responsibilities, dependency direction, context contract, how to add a model.
- `tests/AGENTS.md`: test conventions, markers, fixtures, mocking rules.
- `docs/decisions.md`: dated log of decisions and their reasons.
- `docs/research-report.md`: measured BoW vs TF-IDF results, method and limitations.
- `docs/known-issues.md`: current known oddities, not yet fixed.

## Do not read whole

- `yelp-reviews-project-original.py`: legacy notebook export of about 144 KB, imported nowhere, reference only. Grep it instead.
- `uv.lock`: large generated lockfile. Grep it instead.

## Instruction files

- Keep each fact in exactly one file; put component details in that directory's `AGENTS.md` and link to them from here.
- Do not add `CLAUDE.md`, `CLAUDE.local.md`, `.claude/CLAUDE.md` or `.claude/rules` here; they would suppress native `AGENTS.md` loading.
- Keep each instruction file under 200 lines.
- Update the docs in the same change as the code.
