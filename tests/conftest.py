"""
Shared pytest fixtures for the NLP project test suite.

These fixtures provide reusable test data and mock objects for unit testing
across all modules.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from src.config import DataConfig, ExperimentConfig, ModelConfig, PreprocessingConfig

# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample review texts for testing."""
    return [
        "This restaurant is absolutely amazing! Best food ever.",
        "Terrible service, will never come back again.",
        "It was okay, nothing special but not bad either.",
        "Great atmosphere and friendly staff. Highly recommend!",
        "The food was cold and the waiter was rude.",
        "Average experience, met my basic expectations.",
        "Outstanding quality! A perfect dining experience.",
        "Disappointing. Not worth the price at all.",
        "Pretty good overall, would visit again.",
        "Worst meal I've had in years. Avoid this place.",
    ]


@pytest.fixture
def sample_labels() -> list[int]:
    """Sample labels corresponding to sample_texts (0-4 star ratings)."""
    return [4, 0, 2, 4, 0, 2, 4, 1, 3, 0]


@pytest.fixture
def small_train_texts() -> list[str]:
    """Smaller training set for fast unit tests."""
    return [
        "excellent amazing wonderful great perfect",  # 4
        "terrible horrible awful bad worst",  # 0
        "okay average decent acceptable normal",  # 2
        "fantastic incredible outstanding superb brilliant",  # 4
        "disgusting nasty poor disappointing dreadful",  # 0
        "good nice pleasant enjoyable satisfying",  # 3
        "bad mediocre subpar lacking wanting",  # 1
    ] * 20  # Repeat to have enough samples for vectorizer min_df


@pytest.fixture
def small_train_labels() -> list[int]:
    """Labels for small_train_texts (includes all 5 classes)."""
    return [4, 0, 2, 4, 0, 3, 1] * 20


@pytest.fixture
def small_test_texts() -> list[str]:
    """Small test set for quick evaluation."""
    return [
        "this was a great experience",
        "terrible food never again",
        "it was okay nothing special",
    ]


@pytest.fixture
def small_test_labels() -> list[int]:
    """Labels for small_test_texts."""
    return [4, 0, 2]


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def default_data_config() -> DataConfig:
    """Default DataConfig for testing."""
    return DataConfig(
        dataset_name="yelp_review_full",
        train_split="train",
        test_split="test",
        sample_size=100,  # Small sample for fast tests
    )


@pytest.fixture
def default_preprocessing_config() -> PreprocessingConfig:
    """Default PreprocessingConfig for testing."""
    return PreprocessingConfig(
        min_df=2,  # Lower threshold for small test datasets
        max_df=0.95,
        remove_stopwords=True,
        expand_contractions=True,
        lowercase=True,
    )


@pytest.fixture
def default_model_config() -> ModelConfig:
    """Default ModelConfig for testing."""
    return ModelConfig(
        model_type="bow",
        random_seed=42,
        num_classes=5,
        max_iter=100,  # Fewer iterations for fast tests
    )


@pytest.fixture
def default_experiment_config(
    default_data_config, default_preprocessing_config, default_model_config
) -> ExperimentConfig:
    """Complete ExperimentConfig for testing."""
    return ExperimentConfig(
        data=default_data_config,
        preprocessing=default_preprocessing_config,
        model=default_model_config,
        experiment_name="test_experiment",
    )


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Mock HuggingFace dataset for testing without network calls."""
    mock = MagicMock()
    mock.__getitem__ = MagicMock(
        side_effect=lambda key: {
            "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
            "label": [0, 2, 4],
        }[key]
    )
    return mock


@pytest.fixture
def mock_strategy() -> MagicMock:
    """Mock ModelStrategy for testing pipeline steps."""
    mock = MagicMock()
    mock.train = MagicMock()
    mock.predict = MagicMock(return_value=np.array([0, 1, 2]))
    mock.get_name = MagicMock(return_value="Mock Strategy")
    return mock


# =============================================================================
# Numpy Array Fixtures
# =============================================================================


@pytest.fixture
def sample_predictions() -> NDArray[np.int_]:
    """Sample predictions array for evaluation testing."""
    return np.array([4, 0, 2, 4, 1, 2, 4, 0, 3, 0])


@pytest.fixture
def sample_confusion_matrix() -> NDArray[np.int_]:
    """Sample 5x5 confusion matrix for visualization testing."""
    return np.array(
        [
            [10, 2, 0, 0, 0],
            [1, 8, 3, 0, 0],
            [0, 2, 7, 2, 1],
            [0, 0, 2, 9, 1],
            [0, 0, 1, 2, 9],
        ]
    )
