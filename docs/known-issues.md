# Known issues

Verified against the code at the time of writing. None of these is fixed yet; remove an entry when it is. This file is the only place these issues are described; other files link here. Code is referenced by file and symbol, not line number.

## Dependencies (`pyproject.toml`)

- `torch` is declared only in the optional `deep` extra and `tensorflow` is not declared at all; both are guarded imports in `src/utils/reproducibility.py` `configure_environment`, and seeding is skipped for whichever is absent.

## Behavior

- The five-star default class names remain as fallback in `src/evaluation/evaluator.py`, `src/evaluation/visualizer.py` and when the pipeline context has no `class_names`; the class count itself now comes from the label task.
- `src/utils/reproducibility.py` `configure_environment` overwrites existing `PYTHONHASHSEED`, `MPLCONFIGDIR` and `TF_CPP_MIN_LOG_LEVEL` values. Setting `PYTHONHASHSEED` at runtime does not change hash randomization of the current process.
- `src/models/base.py` (`BaseModel`) and `src/models/adapters.py` (`SklearnModelAdapter`): exported from `src/models/__init__.py` and tested, but not used by the pipeline or factory.
- `src/data/loader.py` `DataLoader._extract_texts_labels`: `sample_size` takes the first N rows, not a random sample; on label-sorted datasets such as DBpedia a small `sample_size` yields a single class, which `TrainModelStep` rejects with a clear error.

## Preprocessing and data

- `src/preprocessing/text_preprocessor.py` `TextPreprocessor`: used only by the LSTM path. It follows the notebook order, so punctuation (including the apostrophe) is replaced before contractions are expanded and apostrophe contractions are not expanded. The classical models use sklearn's lowercase, English `stop_words` and default token pattern inside the vectorizers; see [research-report.md](research-report.md).
- `src/config.py` `PreprocessingConfig`: `expand_contractions` is not read by the vectorizer path in `src/models/strategies.py` (only by the LSTM path); `remove_stopwords` is read there but selects sklearn's built-in English list, not the NLTK stopwords that `TextPreprocessor` uses.
- `src/utils/reproducibility.py` `configure_environment` calls `warnings.simplefilter("ignore", category=Warning)`, which suppresses all warnings, including sklearn convergence warnings.
- Dataset text contains literal backslash-n sequences (25130 of 50000 test rows in the 2026-10-31 run); nothing cleans them, so tokens such as `nworst` enter the vocabulary.

## Configuration and tooling

- `.env.example` sets empty `TF_CPP_MIN_LOG_LEVEL`, `PYTHONHASHSEED` and `MPLCONFIGDIR`, plus `WANDB_DISABLED` and `HF_TOKEN`; no project code reads `HF_TOKEN` or `WANDB_DISABLED` (huggingface_hub reads `HF_TOKEN` from the environment). Nothing loads a `.env` file.
