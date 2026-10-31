# Source package guide

## Module responsibilities and patterns

- `config.py`: settings as dataclasses, passed into constructors (config dependency injection). `ExperimentConfig` classmethods act as presets.
- `data/loader.py`: `DataLoader` reads the HF dataset lazily per split and caches it, reads `DataConfig.text_field`/`label_field`, applies the label task to every split before first-N sampling, and exposes the task's `num_classes` and `class_names`; `sample_size` caveat in [known issues](../docs/known-issues.md#behavior). `data/tasks.py`: `LabelTask` registry (`stars5` identity, `stars3`, `polarity`, `polarity_no_neutral` which drops label 2, `dbpedia14` whose names come from the dataset features); a raw label missing from a task's mapping is dropped. `data/analyzer.py`: `DatasetAnalyzer` computes statistics (punctuation deleted, not spaced; optional stopwords for the vocabulary minus stopwords).
- `preprocessing/text_preprocessor.py`: `TextPreprocessor` cleans text in the notebook's order (lowercase, punctuation to spaces, contractions, `nltk.word_tokenize`, NLTK stopwords, join); the LSTM strategy uses it, NLTK data is fetched lazily. `PUNCTUATION_PATTERN` is the notebook's unescaped `'[' + string.punctuation + ']'`, which leaves the backslash. `preprocessing/feature_engineer.py`: `VectorizerFactory` (Factory) builds BoW and TF-IDF vectorizers; never instantiate sklearn vectorizers directly. `delete_punctuation` there implements the optional `strip_punctuation` cleaning with that same pattern.
- `models/strategies.py`: `ModelStrategy` ABC (Strategy) with `train`, `predict`, `get_name`; implementations `BowLogisticStrategy` and `TfidfLogisticStrategy` (shared `_LogisticStrategy` base), which also expose vocabulary size and top features (classes from the fitted model; a binary model uses the negated coefficients for class 0). `models/factory.py`: `ModelFactory` (Factory) with a class-level registry and `register_lazy(name, "module:Class")` for strategies whose module needs an optional extra: it is imported only in `create`, and its `ImportError` should name the extra. Lazy strategies are built with `n_classes` (the experiment task's class count). `models/base.py` and `models/adapters.py`: `BaseModel` and `SklearnModelAdapter` (Adapter); see [known issues](../docs/known-issues.md#behavior) for their status.
- `pipeline/steps.py`: `PipelineStep` ABC with `execute(context)` and `get_name`; four concrete steps. `pipeline/runner.py`: `run_experiment(name, sample_size)` (load, train, evaluate, summary, top-10 features, plots) and `run_analysis(sample_size)` used by the CLI. `pipeline/workflow.py`: `Workflow` simply runs the steps in order on a copy of the initial context; `WorkflowBuilder` (Builder) is fluent and `build()` raises `ValueError` when empty.
- `evaluation/evaluator.py`: `ModelEvaluator` returns an `EvaluationMetrics` object. `evaluation/visualizer.py`: `ResultsVisualizer` prints summaries and draws plots.
- `utils/reproducibility.py`: `configure_environment(seed)` seeds random and numpy, seeds torch and tensorflow only if importable, and sets `PYTHONHASHSEED`, `MPLCONFIGDIR` (a `configs` folder under the working directory) and `TF_CPP_MIN_LOG_LEVEL`; caveats in [known issues](../docs/known-issues.md#behavior). It also silences warnings by default. `get_seed()` returns the last seed. `main` calls it first with seed 42.

## Dependency direction

- `config` is imported by every other package and imports nothing from the project.
- `pipeline.steps` depends on `data`, `models.factory` and `evaluation`.
- `models.strategies` depends on `preprocessing.feature_engineer`; `models.factory` on `data.tasks` (class count); `models.deep.lstm_glove` on `preprocessing.text_preprocessor`.
- `data.loader` depends on `config` and `data.tasks`.
- `main` depends on `config`, `pipeline`, `evaluation.visualizer` and `utils`.
- Keep this direction; lower layers must not import `pipeline` or `main`.

## Pipeline context contract

Steps share one dict. `Workflow.run` copies the initial context, so callers keep their own dict; a step may also be called directly on a dict.

- `LoadDataStep` writes `train_texts`, `train_labels`, `test_texts`, `test_labels` (task labels) and `class_names`.
- `TrainModelStep` requires and reads `train_texts`, `train_labels`, `test_texts` (missing key raises `KeyError`) and writes `trained_strategy` and `predictions`.
- `EvaluateModelStep` requires `test_labels` and `predictions` (`KeyError` otherwise) and writes `metrics`.
- `VisualizeResultsStep` requires `metrics`, optionally reads `trained_strategy` for the title; this step and `EvaluateModelStep` take class names from the constructor, else context `class_names`, else the five Yelp stars, and leaves the context unchanged.
- The baseline mode reuses loaded data by seeding a second workflow with the four data keys.

## Configuration

- `PreprocessingConfig`: `min_df` 50, `max_df` 0.5 (TF-IDF only; BoW omits it as in the notebook), stopword removal, contraction expansion and lowercasing on, `strip_punctuation` off.
- `ModelConfig`: `model_type` "bow", `random_seed` 42, `max_iter` 100 (sklearn default, the notebook value), `class_weight` None, `embedding_source` (LSTM word vectors: gensim downloader name, default `glove-wiki-gigaword-300`, or a local `.kv`/GloVe text path). The class count is not configured here; it comes from the label task.
- `DataConfig`: dataset `Yelp/yelp_review_full`, splits `train` and `test`, `sample_size` None (full data), `task` "stars5", `text_field` "text", `label_field` "label".
- `ExperimentConfig` bundles the three plus `experiment_name` and `requires_extra`. Presets: `for_bow_baseline`, `for_tfidf`; `for_model(model_type)` picks one by name and raises `ValueError` otherwise.
- Named experiments: the `EXPERIMENTS` dict in `config.py` is the single source (`ExperimentConfig.named`, `experiment_names`); the CLI lists and runs them. To add one, add an entry; if it needs an optional extra, set `requires_extra`, register its strategy with `ModelFactory.register_lazy`, and keep torch/transformers imports inside the strategy module so CLI start-up never imports them.

## Deep models (`models/deep/`, optional `deep` extra)

- Nothing may import `src.models.deep` or `torch` at module level outside this package; register strategies with `ModelFactory.register_lazy("name", "src.models.deep.<module>:Class")` and set `requires_extra="deep"` on the experiment. `training.py` raises an `ImportError` naming `uv sync --extra deep`; `run_experiment` builds the strategy first, so the error appears before any download.
- `training.py` is the shared loop: `TrainingConfig` (lr, batch size, max epochs, `validation_split`, `shuffle_before_split`, `monitor` `val_accuracy`|`val_loss`, `patience`, `restore_best_weights`, optional `reduce_lr_patience` (Keras semantics, at least 1)/`reduce_lr_factor`/`reduce_lr_min_delta` absolute), `fit(model, inputs, labels, config, seed)` returning a per-epoch history, `predict_classes`, `EarlyStopping`. Validation is the last fraction (Keras `validation_split`), of a seeded shuffle unless `shuffle_before_split` is off; the model maps a batch of integer inputs to logits; device is CUDA if available.
- Extensions for transformers, all off by default (char-CNN and LSTM unchanged): `inputs` may be any object indexable by an index tensor that yields a tensor or a dict of tensors; `validation=(inputs, labels)` replaces the internal split; `optimizer` `adam`|`adamw`, `weight_decay` (not applied to parameters with fewer than two dimensions, like HF Trainer), `linear_decay` (lr to 0 over `max_epochs` x steps, stepped every iteration, no warmup), `max_grad_norm`, `evaluate_train` (per-epoch `train_eval_accuracy`). `fit` prints the device.
- Strategies encode their own inputs, seed torch from `ModelConfig.random_seed`, take the class count from the constructor's `n_classes` (the task's, never the training labels) and use the same encoder in `predict`.
- `char_cnn.py`: `CharCnnStrategy` (70-char alphabet, lowercased) and `CharCnnMixedCaseStrategy` (96 chars), experiments `cnn-char` and `cnn-char-mixed-case`.
- `lstm_glove.py`: `LstmGloveStrategy` (cleaning with `TextPreprocessor`, vocabulary of words with count >= 50, OOV to 0, post padding/truncation to 300, frozen GloVe embedding matrix, `BiLstmClassifier` with final states of both directions, no masking, Keras-like init). Experiments `lstm-glove` (`stars5`) and `lstm-glove-3class` (`stars3`) use `notebook_training_config()` and restore the best `val_loss` weights. GloVe loads (via `gensim`) only in `train`. PyTorch's extra LSTM bias makes 1,570,309 trainable parameters for five classes against Keras's 1,568,261 ([decisions](../docs/decisions.md)).

- `distilbert.py`: `TransformerStrategy` base (injectable `tokenizer`/`model` for offline tests; a future no-training strategy such as zero-shot overrides `train`), `TokenizedTexts` (lazy `max_length` or dynamic padding, truncation 512), `TransformerClassifier` (batch dict to logits), `freeze_encoder`, `parameter_summary`. `DistilBertFineTuneStrategy` (`distilbert-finetune`) and `DistilBertTransferStrategy` (`distilbert-transfer`, `distilbert` encoder frozen) use `five_class_training_config()`: AdamW 5e-5, batch 8, 1 epoch, linear decay, seeded shuffle then 0.75/0.25 split, early stopping patience 2, train-part evaluation per epoch (`evaluate_train=True`, as the notebook; about a third more compute on a full run, set it to `False` in `five_class_training_config()` to skip it; the history then has no `train_eval_accuracy`). `TokenizedTexts` tokenizes 10,000 texts at a time and keeps int32 rows. `DistilBertPolarityStrategy` (`distilbert-polarity`, task `polarity`, `strip_punctuation` on) splits with `train_test_split(test_size=0.2, random_state=2307)` and uses `polarity_training_config()`: batch 32, lr 2e-5, weight decay 0.01, clip 1.0, dynamic padding. No hub access or tokens anywhere.

## Adding a model

1. Subclass `ModelStrategy` in `models/strategies.py` (or a new module) implementing `train`, `predict`, `get_name`; take the model and preprocessing configs in the constructor.
2. Build features through `VectorizerFactory`; add a creator there if a new vectorizer is needed.
3. Register the class in `ModelFactory` (`ModelFactory.register(name, cls)` or the registry in `models/factory.py`) and add its preset to `ExperimentConfig.for_model`; `single <name>` picks it up.
4. Add tests in `tests/test_models.py` and `tests/test_config.py`, and update the docs.

## Rules

- CLI input errors fail loudly (message and exit 1), never silent fallbacks.
- Keep no presets or config for a model without a registered strategy.
- Hugging Face dataset ids use the canonical `namespace/name` form; huggingface-hub 2.x rejects bare legacy ids.

## Reproducibility

Call `configure_environment` before loading data or training. Model seeds come from `ModelConfig.random_seed`.
