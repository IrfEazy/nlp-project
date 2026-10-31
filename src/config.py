"""
Configuration dataclasses for the NLP pipeline.

All settings are centralized in type-safe dataclasses following the
Single Responsibility Principle. Pass config objects to constructors
for Dependency Injection rather than hardcoding values.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing and vectorization.

    Attributes:
        min_df: Minimum document frequency for vocabulary filtering.
                Words appearing in fewer documents are excluded.
        max_df: Maximum document frequency (0.0-1.0). Words in more than
                this fraction of documents are excluded.
        remove_stopwords: Whether to remove English stopwords.
                         Set True for traditional ML, False for transformers.
        expand_contractions: Whether to expand contractions (e.g., "that's" -> "that is").
        lowercase: Whether to convert text to lowercase.
        strip_punctuation: Whether to delete ``string.punctuation`` characters
                (replaced by '') before vectorizing. Off by default: the
                sklearn vectorizers get raw text.
    """

    min_df: int = 50
    max_df: float = 0.5
    remove_stopwords: bool = True
    expand_contractions: bool = True
    lowercase: bool = True
    strip_punctuation: bool = False


@dataclass
class ModelConfig:
    """Configuration for model training.

    Attributes:
        model_type: Type of model to use ('bow' or 'tfidf').
        random_seed: Seed for reproducibility.
        max_iter: Maximum iterations for sklearn models (sklearn default).
        class_weight: Passed to LogisticRegression (None, 'balanced' or a dict).
    The class count comes from the label task (``DataConfig.task``).
    """

    model_type: str = "bow"
    random_seed: int = 42
    max_iter: int = 100
    class_weight: Union[None, str, Dict[int, float]] = None


@dataclass
class DataConfig:
    """Configuration for data loading.

    Attributes:
        dataset_name: HuggingFace dataset identifier.
        train_split: Name of the training split.
        test_split: Name of the test split.
        sample_size: Optional limit on samples to load (None for full dataset).
        task: Label task name from ``src.data.tasks`` (default 'stars5').
        text_field: Dataset column holding the text.
        label_field: Dataset column holding the label.
    """

    dataset_name: str = "Yelp/yelp_review_full"
    train_split: str = "train"
    test_split: str = "test"
    sample_size: Optional[int] = None
    task: str = "stars5"
    text_field: str = "text"
    label_field: str = "label"


@dataclass
class ExperimentConfig:
    """Bundle of all configuration for a complete experiment.

    Attributes:
        data: Data loading configuration.
        preprocessing: Text preprocessing configuration.
        model: Model training configuration.
        experiment_name: Human-readable name for the experiment.
        requires_extra: Optional dependency extra the experiment needs
                (e.g. 'deep'); informational, checked when the model is built.
    """

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment_name: str = "default_experiment"
    requires_extra: Optional[str] = None

    @classmethod
    def for_bow_baseline(cls) -> "ExperimentConfig":
        """Create configuration for Bag-of-Words baseline."""
        return cls(
            data=DataConfig(),
            preprocessing=PreprocessingConfig(
                min_df=50,
                remove_stopwords=True,
            ),
            model=ModelConfig(model_type="bow"),
            experiment_name="bow_baseline",
        )

    @classmethod
    def for_tfidf(cls) -> "ExperimentConfig":
        """Create configuration for TF-IDF model."""
        return cls(
            data=DataConfig(),
            preprocessing=PreprocessingConfig(
                min_df=50,
                max_df=0.5,
                remove_stopwords=True,
            ),
            model=ModelConfig(model_type="tfidf"),
            experiment_name="tfidf_logistic",
        )

    @classmethod
    def for_model(cls, model_type: str) -> "ExperimentConfig":
        """Return the preset for a model type.

        Args:
            model_type: Model type ('bow' or 'tfidf').

        Raises:
            ValueError: If no preset exists for the model type.
        """
        presets = {"bow": cls.for_bow_baseline, "tfidf": cls.for_tfidf}
        if model_type not in presets:
            raise ValueError(
                f"No preset for model type '{model_type}'. "
                f"Available presets: {sorted(presets)}"
            )
        return presets[model_type]()

    @classmethod
    def named(cls, name: str) -> "ExperimentConfig":
        """Return a fresh config for a registered experiment name.

        Raises:
            ValueError: If the name is not in ``EXPERIMENTS``.
        """
        if name not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment '{name}'. "
                f"Available experiments: {', '.join(EXPERIMENTS)}"
            )
        return EXPERIMENTS[name]()

    @staticmethod
    def experiment_names() -> List[str]:
        """Return the registered experiment names."""
        return list(EXPERIMENTS)


def _lr_experiment(
    name: str,
    model_type: str,
    data: Optional[DataConfig] = None,
    class_weight: Union[None, str, Dict[int, float]] = None,
    strip_punctuation: bool = False,
) -> Callable[[], ExperimentConfig]:
    """Build a preset factory for a logistic regression experiment."""

    def build() -> ExperimentConfig:
        return ExperimentConfig(
            data=DataConfig(**vars(data)) if data else DataConfig(),
            preprocessing=PreprocessingConfig(
                min_df=50, max_df=0.5, strip_punctuation=strip_punctuation
            ),
            model=ModelConfig(model_type=model_type, class_weight=class_weight),
            experiment_name=name,
        )

    return build


def _deep_experiment(
    name: str, model_type: str, data: Optional[DataConfig] = None
) -> Callable[[], ExperimentConfig]:
    """Build a preset factory for a deep-learning experiment (needs the 'deep' extra)."""

    def build() -> ExperimentConfig:
        return ExperimentConfig(
            data=DataConfig(**vars(data)) if data else DataConfig(),
            model=ModelConfig(model_type=model_type),
            experiment_name=name,
            requires_extra="deep",
        )

    return build


# Single source of named experiments (the original notebook's classical runs).
# Later packages add entries here; entries that need an optional extra set
# ExperimentConfig.requires_extra and keep heavy imports out of this module.
EXPERIMENTS: Dict[str, Callable[[], ExperimentConfig]] = {
    "lr-bow": _lr_experiment("lr-bow", "bow"),
    "lr-tfidf": _lr_experiment("lr-tfidf", "tfidf"),
    "lr-bow-3class": _lr_experiment(
        "lr-bow-3class", "bow", DataConfig(task="stars3"), class_weight="balanced"
    ),
    "lr-bow-polarity": _lr_experiment(
        "lr-bow-polarity",
        "bow",
        DataConfig(task="polarity"),
        class_weight="balanced",
        strip_punctuation=True,
    ),
    "lr-bow-no-neutral": _lr_experiment(
        "lr-bow-no-neutral", "bow", DataConfig(task="polarity_no_neutral")
    ),
    "lr-bow-dbpedia": _lr_experiment(
        "lr-bow-dbpedia",
        "bow",
        DataConfig(dataset_name="fancyzhx/dbpedia_14", task="dbpedia14", text_field="content"),
    ),
    "cnn-char": _deep_experiment("cnn-char", "cnn-char"),
    "cnn-char-mixed-case": _deep_experiment("cnn-char-mixed-case", "cnn-char-mixed-case"),
}
