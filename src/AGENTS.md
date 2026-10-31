# Source package guide

## Module responsibilities and patterns

- `config.py`: settings as dataclasses, passed into constructors (config dependency injection). `ExperimentConfig` classmethods act as presets.
- `data/loader.py`: `DataLoader` reads the HF dataset lazily per split and caches it. `sample_size` keeps the first N rows (head slice, not random). `data/analyzer.py`: `DatasetAnalyzer` computes statistics.
- `preprocessing/text_preprocessor.py`: `TextPreprocessor` cleans text (contractions, punctuation, NLTK stopwords). `preprocessing/feature_engineer.py`: `VectorizerFactory` (Factory) builds BoW and TF-IDF vectorizers; never instantiate sklearn vectorizers directly.
- `models/strategies.py`: `ModelStrategy` ABC (Strategy) with `train`, `predict`, `get_name`; implementations `BowLogisticStrategy` and `TfidfLogisticStrategy`, which also expose vocabulary size and top features. `models/factory.py`: `ModelFactory` (Factory) with a class-level registry. `models/base.py` and `models/adapters.py`: `BaseModel` and `SklearnModelAdapter` (Adapter); see [known issues](../docs/known-issues.md#behavior) for their status.
- `pipeline/steps.py`: `PipelineStep` ABC with `execute(context)` and `get_name`; four concrete steps. `pipeline/workflow.py`: `Workflow` simply runs the steps in order on a copy of the initial context; `WorkflowBuilder` (Builder) is fluent and `build()` raises `ValueError` when empty.
- `evaluation/evaluator.py`: `ModelEvaluator` returns an `EvaluationMetrics` object. `evaluation/visualizer.py`: `ResultsVisualizer` prints summaries and draws plots.
- `utils/reproducibility.py`: `configure_environment(seed)` seeds random and numpy, seeds torch and tensorflow only if importable, and sets `PYTHONHASHSEED`, `MPLCONFIGDIR` (a `configs` folder under the working directory) and `TF_CPP_MIN_LOG_LEVEL`; caveats in [known issues](../docs/known-issues.md#behavior). It also silences warnings by default. `get_seed()` returns the last seed. `main` calls it first with seed 42.

## Dependency direction

- `config` is imported by every other package and imports nothing from the project.
- `pipeline.steps` depends on `data`, `models.factory` and `evaluation`.
- `models.strategies` depends on `preprocessing.feature_engineer`.
- `main` depends on `config`, `pipeline`, `evaluation.visualizer` and `utils`.
- Keep this direction; lower layers must not import `pipeline` or `main`.

## Pipeline context contract

Steps share one dict. `Workflow.run` copies the initial context, so callers keep their own dict; a step may also be called directly on a dict.

- `LoadDataStep` writes `train_texts`, `train_labels`, `test_texts`, `test_labels`.
- `TrainModelStep` requires and reads `train_texts`, `train_labels`, `test_texts` (missing key raises `KeyError`) and writes `trained_strategy` and `predictions`.
- `EvaluateModelStep` requires `test_labels` and `predictions` (`KeyError` otherwise) and writes `metrics`.
- `VisualizeResultsStep` requires `metrics`, optionally reads `trained_strategy` for the title, and leaves the context unchanged.
- The baseline mode reuses loaded data by seeding a second workflow with the four data keys.

## Configuration

- `PreprocessingConfig`: `min_df` 50, `max_df` 0.5, stopword removal, contraction expansion and lowercasing all on.
- `ModelConfig`: `model_type` "bow", `random_seed` 42, `num_classes` 5, `max_iter` 1000.
- `DataConfig`: dataset `Yelp/yelp_review_full`, splits `train` and `test`, `sample_size` None (full data).
- `ExperimentConfig` bundles the three plus `experiment_name`. Presets: `for_bow_baseline`, `for_tfidf`; `for_model(model_type)` picks one by name and raises `ValueError` otherwise.

## Adding a model

1. Subclass `ModelStrategy` in `models/strategies.py` (or a new module) implementing `train`, `predict`, `get_name`; take the model and preprocessing configs in the constructor.
2. Build features through `VectorizerFactory`; add a creator there if a new vectorizer is needed.
3. Register the class in `ModelFactory` (`ModelFactory.register(name, cls)` or the registry in `models/factory.py`) and add its preset to `ExperimentConfig.for_model`; `single <name>` picks it up.
4. Add tests in `tests/test_models.py` and `tests/test_config.py`, and update the docs.

## Reproducibility

Call `configure_environment` before loading data or training. Model seeds come from `ModelConfig.random_seed`.
