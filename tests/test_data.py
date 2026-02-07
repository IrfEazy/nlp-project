"""
Tests for the data module.

Tests cover:
- DataLoader with mocked dataset
- DatasetAnalyzer statistics computation
"""

from unittest.mock import patch

import pytest

from src.config import DataConfig
from src.data.analyzer import DatasetAnalyzer, DatasetStats
from src.data.loader import DataLoader


@pytest.mark.unit
class TestDataLoader:
    """Tests for DataLoader class."""

    def test_initialization(self, default_data_config) -> None:
        """DataLoader should initialize with config."""
        loader = DataLoader(default_data_config)

        assert loader.config == default_data_config
        assert loader._train_dataset is None
        assert loader._test_dataset is None

    @patch("src.data.loader.load_dataset")
    def test_load_train_calls_load_dataset(
        self, mock_load_dataset, default_data_config
    ) -> None:
        """load_train should call load_dataset with correct parameters."""
        mock_dataset = {"text": ["text1", "text2", "text3"], "label": [0, 2, 4]}
        mock_load_dataset.return_value = mock_dataset

        loader = DataLoader(default_data_config)
        texts, labels = loader.load_train()

        mock_load_dataset.assert_called_once_with(
            default_data_config.dataset_name, split=default_data_config.train_split
        )

    @patch("src.data.loader.load_dataset")
    def test_load_train_returns_lists(
        self, mock_load_dataset, default_data_config
    ) -> None:
        """load_train should return lists of texts and labels."""
        mock_dataset = {"text": ["text1", "text2", "text3"], "label": [0, 2, 4]}
        mock_load_dataset.return_value = mock_dataset

        loader = DataLoader(default_data_config)
        texts, labels = loader.load_train()

        assert isinstance(texts, list)
        assert isinstance(labels, list)
        assert texts == ["text1", "text2", "text3"]
        assert labels == [0, 2, 4]

    @patch("src.data.loader.load_dataset")
    def test_load_train_applies_sampling(self, mock_load_dataset) -> None:
        """load_train should apply sample_size limit."""
        mock_dataset = {
            "text": ["text1", "text2", "text3", "text4", "text5"],
            "label": [0, 1, 2, 3, 4],
        }
        mock_load_dataset.return_value = mock_dataset

        config = DataConfig(sample_size=3)
        loader = DataLoader(config)
        texts, labels = loader.load_train()

        assert len(texts) == 3
        assert len(labels) == 3

    @patch("src.data.loader.load_dataset")
    def test_load_test_separate_from_train(
        self, mock_load_dataset, default_data_config
    ) -> None:
        """load_test should load test split separately."""
        mock_load_dataset.return_value = {"text": ["test_text"], "label": [1]}

        loader = DataLoader(default_data_config)
        texts, labels = loader.load_test()

        # Check it called load_dataset with test split
        mock_load_dataset.assert_called_with(
            default_data_config.dataset_name, split=default_data_config.test_split
        )

    @patch("src.data.loader.load_dataset")
    def test_caching_train_dataset(
        self, mock_load_dataset, default_data_config
    ) -> None:
        """Subsequent load_train calls should not re-load dataset."""
        mock_dataset = {"text": ["text1"], "label": [0]}
        mock_load_dataset.return_value = mock_dataset

        loader = DataLoader(default_data_config)

        # First call
        loader.load_train()
        # Second call
        loader.load_train()

        # Should only call load_dataset once
        assert mock_load_dataset.call_count == 1

    def test_num_classes(self, default_data_config) -> None:
        """num_classes should return 5 for Yelp dataset."""
        loader = DataLoader(default_data_config)
        assert loader.num_classes == 5

    def test_class_names(self, default_data_config) -> None:
        """class_names should return list of string labels."""
        loader = DataLoader(default_data_config)
        names = loader.class_names

        assert names == ["0", "1", "2", "3", "4"]


@pytest.mark.unit
class TestDatasetAnalyzer:
    """Tests for DatasetAnalyzer class."""

    def test_initialization_default(self) -> None:
        """Analyzer should initialize with default settings."""
        analyzer = DatasetAnalyzer()
        assert analyzer.remove_punctuation

    def test_initialization_no_punctuation_removal(self) -> None:
        """Analyzer can be configured to not remove punctuation."""
        analyzer = DatasetAnalyzer(remove_punctuation=False)
        assert not analyzer.remove_punctuation

    def test_compute_stats_empty_list(self) -> None:
        """compute_stats should handle empty list."""
        analyzer = DatasetAnalyzer()
        stats = analyzer.compute_stats([])

        assert isinstance(stats, DatasetStats)
        assert stats.num_documents == 0
        assert stats.avg_document_length == 0.0

    def test_compute_stats_returns_dataclass(self, sample_texts) -> None:
        """compute_stats should return DatasetStats dataclass."""
        analyzer = DatasetAnalyzer()
        stats = analyzer.compute_stats(sample_texts)

        assert isinstance(stats, DatasetStats)

    def test_compute_stats_document_count(self, sample_texts) -> None:
        """compute_stats should count documents correctly."""
        analyzer = DatasetAnalyzer()
        stats = analyzer.compute_stats(sample_texts)

        assert stats.num_documents == len(sample_texts)

    def test_compute_stats_avg_length(self) -> None:
        """compute_stats should compute average document length."""
        texts = ["hello", "hello world", "hi"]  # lengths: 5, 11, 2 = avg 6
        analyzer = DatasetAnalyzer()
        stats = analyzer.compute_stats(texts)

        assert stats.avg_document_length == 6.0

    def test_compute_stats_vocabulary(self) -> None:
        """compute_stats should compute vocabulary statistics."""
        texts = ["hello world", "hello there", "world peace"]
        analyzer = DatasetAnalyzer()
        stats = analyzer.compute_stats(texts)

        # Total unique words: hello, world, there, peace = 4
        assert stats.total_vocabulary == 4
        # Avg per doc: 2, 2, 2 = 2
        assert stats.avg_vocabulary_per_doc == 2.0

    def test_compute_avg_document_length(self, sample_texts) -> None:
        """compute_avg_document_length should return float."""
        analyzer = DatasetAnalyzer()
        avg_len = analyzer.compute_avg_document_length(sample_texts)

        assert isinstance(avg_len, float)
        assert avg_len > 0

    def test_compute_avg_document_length_empty(self) -> None:
        """compute_avg_document_length should handle empty list."""
        analyzer = DatasetAnalyzer()
        avg_len = analyzer.compute_avg_document_length([])

        assert avg_len == 0.0

    def test_get_label_distribution(self, sample_labels) -> None:
        """get_label_distribution should count labels correctly."""
        analyzer = DatasetAnalyzer()
        distribution = analyzer.get_label_distribution(sample_labels)

        assert isinstance(distribution, dict)
        # sample_labels = [4, 0, 2, 4, 0, 2, 4, 1, 3, 0]
        assert distribution[0] == 3  # Three 0 labels
        assert distribution[4] == 3  # Three 4 labels
        assert distribution[2] == 2  # Two 2 labels

    def test_punctuation_removal_affects_vocabulary(self) -> None:
        """Punctuation removal should affect vocabulary computation."""
        texts = ["hello, world!", "hello world"]

        with_removal = DatasetAnalyzer(remove_punctuation=True)
        without_removal = DatasetAnalyzer(remove_punctuation=False)

        stats_with = with_removal.compute_stats(texts)
        stats_without = without_removal.compute_stats(texts)

        # With removal: "hello" and "world" (2 words)
        # Without removal: "hello,", "world!", "hello", "world" (4 tokens with punctuation)
        assert stats_with.total_vocabulary == 2
        assert stats_without.total_vocabulary > stats_with.total_vocabulary


@pytest.mark.unit
class TestDatasetStats:
    """Tests for DatasetStats dataclass."""

    def test_dataclass_fields(self) -> None:
        """DatasetStats should have expected fields."""
        stats = DatasetStats(
            num_documents=100,
            avg_document_length=500.5,
            total_vocabulary=10000,
            avg_vocabulary_per_doc=50.5,
        )

        assert stats.num_documents == 100
        assert stats.avg_document_length == 500.5
        assert stats.total_vocabulary == 10000
        assert stats.avg_vocabulary_per_doc == 50.5
        assert stats.avg_vocabulary_per_doc == 50.5
