# Known issues

Verified against the code at the time of writing. None of these is fixed yet; remove an entry when it is. This file is the only place these issues are described; other files link here. Code is referenced by file and symbol, not line number.

## Dependencies (`pyproject.toml`)

- `torch` and `tensorflow` are optional guarded imports in `src/utils/reproducibility.py` `configure_environment`; neither is declared, and seeding is skipped when they are absent.
- `matplotlib` (`src/evaluation/visualizer.py`) and `numpy` are imported by `src/` but are only available transitively (matplotlib via seaborn, numpy via scikit-learn and others); neither is declared in `pyproject.toml`.
- `pytest` and `pytest-cov` are in runtime `dependencies`, not a dev group.
- `description` is still the placeholder text.

## Behavior

- `num_classes` 5 is duplicated: `src/data/loader.py` `DataLoader.num_classes` and `class_names`, `src/config.py` `ModelConfig`, `src/pipeline/steps.py` (default class names in `EvaluateModelStep` and `VisualizeResultsStep`), and `src/evaluation/visualizer.py`.
- `src/utils/reproducibility.py` `configure_environment` overwrites existing `PYTHONHASHSEED`, `MPLCONFIGDIR` and `TF_CPP_MIN_LOG_LEVEL` values. Setting `PYTHONHASHSEED` at runtime does not change hash randomization of the current process.
- `src/models/base.py` (`BaseModel`) and `src/models/adapters.py` (`SklearnModelAdapter`): exported from `src/models/__init__.py` and tested, but not used by the pipeline or factory.
- `src/data/loader.py` `DataLoader._extract_texts_labels`: `sample_size` takes the first N rows, not a random sample.

## Configuration and tooling

- `.env.example` sets empty `TF_CPP_MIN_LOG_LEVEL`, `PYTHONHASHSEED` and `MPLCONFIGDIR`, plus `WANDB_DISABLED` and `HF_TOKEN`, which no code reads. Nothing loads a `.env` file.
- The `slow` pytest marker is registered in `pyproject.toml` but no test uses it.
