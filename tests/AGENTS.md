# Test guide

## Configuration

- Pytest settings live in `pyproject.toml`: tests under `tests/`, files `test_*.py`, classes `Test*`, functions `test_*`.
- Options `--strict-markers -v` are on, so unregistered markers fail.
- Registered markers are only `unit` (fast, isolated) and `integration` (may use external resources).
- Every test class must carry `unit` or `integration`, applied as a class decorator.
- Coverage measures `src` and omits `__init__.py` files.

## Workflow

- New behaviour ships with tests in the matching test file.
- The full suite (`uv run pytest`) must pass before committing.

## Layout

- `test_deep.py` covers `src/models/deep/` (char-CNN, shared loop, BiLSTM/GloVe with a fake word-vector dict and a stubbed tokenizer, so no GloVe, NLTK or dataset download): it calls `pytest.importorskip("torch")` first, so it skips without the `deep` extra, and is the only test file that imports torch directly (besides `configure_environment`). Tiny configs only, no network.
- `test_distilbert.py` covers `models/deep/distilbert.py` and the shared-loop extensions: `importorskip` for torch and transformers, a tiny `DistilBertConfig` model and a `BertTokenizer` built from a throwaway vocab file in a temp dir, injected into the strategies; no checkpoint download.
- `test_zeroshot.py` covers `models/deep/zeroshot.py` and the evaluation-only data path: `importorskip` for torch and transformers, a fake pipeline injected as `classifier` (labels, templates, `multi_label`, batching, label-to-id mapping); the pipeline constructor is patched, so no model download.
- One file per source package: `test_config.py`, `test_data.py`, `test_evaluation.py`, `test_models.py`, `test_pipeline.py`, `test_preprocessing.py`, `test_utils.py`, plus `test_main.py` for the entry point.
- New package means a new `test_<package>.py` here.
- Import project code as `from src.x import ...`.

## Fixtures (`conftest.py`)

- Data: `sample_texts`, `sample_labels`, `small_train_texts`, `small_train_labels`, `small_test_texts`, `small_test_labels`. The small sets repeat texts so vectorizer `min_df` is satisfied and cover all 5 classes.
- Configs: `default_data_config`, `default_preprocessing_config`, `default_model_config`, `default_experiment_config`. Tests for runners and the experiment CLI patch `src.data.loader.load_dataset` or `src.main.run_experiment`/`run_analysis`; stopwords are patched via `src.pipeline.runner._english_stopwords`.
- Mocks: `mock_dataset` (fake HF dataset with `text` and `label`), `mock_strategy` (fake `ModelStrategy`).
- Arrays: `sample_predictions`, `sample_confusion_matrix`.

## Mocking rules

- Patch names where they are used, not where defined:
  - `src.data.loader.load_dataset` for dataset access.
  - `src.pipeline.steps.DataLoader` and `src.pipeline.steps.ModelFactory` for step tests.
  - `src.main.*` for workflow components, `configure_environment` and configs in entry-point tests.
- Never hit the network in tests.
- scikit-learn and NLTK data is never downloaded in tests: the `stub_nltk` conftest fixture replaces tokenization and stopwords, and the `real_nltk` fixture (notebook cleaning-order tests) skips when the punkt/stopwords data is absent; `gensim` is never exercised, GloVe is a dict). Only `test_deep.py` and `test_distilbert.py` import torch directly; tests never import TensorFlow, but `configure_environment` (called unpatched in `test_utils.py`) imports torch and tensorflow when installed; transformers is imported only by `test_distilbert.py` (tiny local model and tokenizer, never `from_pretrained`).
- In `test_evaluation.py`, the `TestResultsVisualizer` class selects the matplotlib `Agg` backend through an autouse fixture.
