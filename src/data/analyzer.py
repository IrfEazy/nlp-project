"""
Dataset analysis utilities.

Provides statistics computation for text datasets including document length,
vocabulary size, and other corpus-level metrics.
"""

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import List, Set


@dataclass
class DatasetStats:
    """Statistics for a text dataset.

    Attributes:
        num_documents: Total number of documents.
        avg_document_length: Average character count per document.
        total_vocabulary: Total unique words in the corpus.
        avg_vocabulary_per_doc: Average unique words per document.
    """

    num_documents: int
    avg_document_length: float
    total_vocabulary: int
    avg_vocabulary_per_doc: float


class DatasetAnalyzer:
    """Computes statistics for text datasets.

    Single Responsibility: Only analyze and compute statistics,
    does NOT load or preprocess data.

    Example:
        >>> analyzer = DatasetAnalyzer()
        >>> stats = analyzer.compute_stats(texts)
        >>> print(f"Avg length: {stats.avg_document_length:.2f}")
    """

    def __init__(self, remove_punctuation: bool = True) -> None:
        """Initialize the analyzer.

        Args:
            remove_punctuation: Whether to remove punctuation when computing
                vocabulary statistics.
        """
        self.remove_punctuation = remove_punctuation
        self._punctuation_pattern = "[" + string.punctuation + "]"

    def compute_stats(self, texts: List[str]) -> DatasetStats:
        """Compute comprehensive statistics for a list of texts.

        Args:
            texts: List of document strings.

        Returns:
            DatasetStats object containing computed statistics.
        """
        num_docs = len(texts)
        if num_docs == 0:
            return DatasetStats(
                num_documents=0,
                avg_document_length=0.0,
                total_vocabulary=0,
                avg_vocabulary_per_doc=0.0,
            )

        total_vocab: Set[str] = set()
        total_length = 0
        total_unique_per_doc = 0

        for text in texts:
            # Document length
            total_length += len(text)

            # Vocabulary
            cleaned = self._clean_for_vocab(text)
            words = cleaned.lower().split()
            unique_words = set(words)

            total_vocab.update(unique_words)
            total_unique_per_doc += len(unique_words)

        return DatasetStats(
            num_documents=num_docs,
            avg_document_length=total_length / num_docs,
            total_vocabulary=len(total_vocab),
            avg_vocabulary_per_doc=total_unique_per_doc / num_docs,
        )

    def _clean_for_vocab(self, text: str) -> str:
        """Clean text for vocabulary counting.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text with punctuation optionally removed.
        """
        if self.remove_punctuation:
            return re.sub(self._punctuation_pattern, " ", text)
        return text

    def compute_avg_document_length(self, texts: List[str]) -> float:
        """Compute average document length in characters.

        Args:
            texts: List of document strings.

        Returns:
            Average character count per document.
        """
        if not texts:
            return 0.0
        return sum(len(text) for text in texts) / len(texts)

    def get_label_distribution(self, labels: List[int]) -> dict:
        """Compute label distribution.

        Args:
            labels: List of integer labels.

        Returns:
            Dictionary mapping label to count.
        """
        return dict(Counter(labels))
