"""
Data loading utilities for the Yelp review dataset.

This module provides the DataLoader class which handles loading data from
HuggingFace datasets with optional sampling for faster iteration.
"""

from typing import List, Tuple

from datasets import load_dataset

from src.config import DataConfig


class DataLoader:
    """Loads and provides access to the Yelp review dataset.

    Handles loading data from HuggingFace, optional sampling, and
    provides a clean interface for accessing texts and labels.

    Attributes:
        config: DataConfig specifying dataset name, splits, and sample size.

    Example:
        >>> config = DataConfig(sample_size=1000)  # For faster iteration
        >>> loader = DataLoader(config)
        >>> train_texts, train_labels = loader.load_train()
        >>> test_texts, test_labels = loader.load_test()
    """

    def __init__(self, config: DataConfig) -> None:
        """Initialize the data loader.

        Args:
            config: Configuration specifying dataset and sampling parameters.
        """
        self.config = config
        self._train_dataset = None
        self._test_dataset = None

    def load_train(self) -> Tuple[List[str], List[int]]:
        """Load training data.

        Returns:
            Tuple of (texts, labels) where texts is a list of review strings
            and labels is a list of integer ratings (0-4).
        """
        if self._train_dataset is None:
            self._train_dataset = load_dataset(
                self.config.dataset_name, split=self.config.train_split
            )

        return self._extract_texts_labels(self._train_dataset)

    def load_test(self) -> Tuple[List[str], List[int]]:
        """Load test data.

        Returns:
            Tuple of (texts, labels) where texts is a list of review strings
            and labels is a list of integer ratings (0-4).
        """
        if self._test_dataset is None:
            self._test_dataset = load_dataset(
                self.config.dataset_name, split=self.config.test_split
            )

        return self._extract_texts_labels(self._test_dataset)

    def _extract_texts_labels(self, dataset) -> Tuple[List[str], List[int]]:
        """Extract texts and labels from a HuggingFace dataset.

        Args:
            dataset: A HuggingFace Dataset object.

        Returns:
            Tuple of (texts, labels) lists, optionally sampled.
        """
        texts = dataset["text"]
        labels = dataset["label"]

        # Apply sampling if configured
        if self.config.sample_size is not None:
            sample_size = min(self.config.sample_size, len(texts))
            texts = texts[:sample_size]
            labels = labels[:sample_size]

        return list(texts), list(labels)

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset (5 for Yelp: 0-4 stars)."""
        return 5

    @property
    def class_names(self) -> List[str]:
        """Human-readable class names."""
        return ["0", "1", "2", "3", "4"]
