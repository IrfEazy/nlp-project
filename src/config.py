"""
Configuration dataclasses for the NLP pipeline.

All settings are centralized in type-safe dataclasses following the
Single Responsibility Principle. Pass config objects to constructors
for Dependency Injection rather than hardcoding values.
"""

from dataclasses import dataclass, field
from typing import Optional


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
    """

    min_df: int = 50
    max_df: float = 0.5
    remove_stopwords: bool = True
    expand_contractions: bool = True
    lowercase: bool = True


@dataclass
class ModelConfig:
    """Configuration for model training.

    Attributes:
        model_type: Type of model to use ('bow', 'tfidf', 'distilbert').
        random_seed: Seed for reproducibility.
        num_classes: Number of output classes (5 for Yelp star ratings).
        max_iter: Maximum iterations for sklearn models.
    """

    model_type: str = "bow"
    random_seed: int = 42
    num_classes: int = 5
    max_iter: int = 1000


@dataclass
class DataConfig:
    """Configuration for data loading.

    Attributes:
        dataset_name: HuggingFace dataset identifier.
        train_split: Name of the training split.
        test_split: Name of the test split.
        sample_size: Optional limit on samples to load (None for full dataset).
    """

    dataset_name: str = "yelp_review_full"
    train_split: str = "train"
    test_split: str = "test"
    sample_size: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Bundle of all configuration for a complete experiment.

    Attributes:
        data: Data loading configuration.
        preprocessing: Text preprocessing configuration.
        model: Model training configuration.
        experiment_name: Human-readable name for the experiment.
    """

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment_name: str = "default_experiment"

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
    def for_distilbert(cls) -> "ExperimentConfig":
        """Create configuration for DistilBERT (transformers don't remove stopwords)."""
        return cls(
            data=DataConfig(),
            preprocessing=PreprocessingConfig(
                remove_stopwords=False,  # Keep stopwords for transformers
                expand_contractions=False,
            ),
            model=ModelConfig(model_type="distilbert"),
            experiment_name="distilbert_finetune",
        )
