"""
Text preprocessing utilities.

Provides text cleaning operations including lowercasing, punctuation removal,
contraction expansion, and stopword removal.
"""

import re
import string
from typing import List, Optional

import contractions
import nltk
from nltk.corpus import stopwords

from src.config import PreprocessingConfig


class TextPreprocessor:
    """Cleans and preprocesses text for NLP tasks.

    Single Responsibility: Only clean text, does NOT vectorize.

    The preprocessing logic differs by model type:
    - Traditional ML (BoW, TF-IDF): remove_stopwords=True to reduce noise
    - Transformers (DistilBERT): remove_stopwords=False because words like
      "not", "but" carry sentiment signal

    Example:
        >>> config = PreprocessingConfig(remove_stopwords=True)
        >>> preprocessor = TextPreprocessor(config)
        >>> cleaned = preprocessor.preprocess("That's a GREAT restaurant!")
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize the preprocessor.

        Args:
            config: Configuration specifying preprocessing options.
        """
        self.config = config
        self._punctuation_pattern = "[" + re.escape(string.punctuation) + "]"
        self._stopwords: Optional[set] = None

        # Lazy load stopwords only if needed
        if config.remove_stopwords:
            self._load_stopwords()

    def _load_stopwords(self) -> None:
        """Load NLTK stopwords (downloads if necessary)."""
        try:
            self._stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self._stopwords = set(stopwords.words("english"))

    def preprocess(self, text: str) -> str:
        """Apply all configured preprocessing steps to a single text.

        Args:
            text: Raw text string.

        Returns:
            Preprocessed text string.
        """
        result = text

        # Lowercase
        if self.config.lowercase:
            result = result.lower()

        # Expand contractions
        if self.config.expand_contractions:
            result = self._expand_contractions(result)

        # Remove punctuation
        result = self._remove_punctuation(result)

        # Remove stopwords
        if self.config.remove_stopwords and self._stopwords:
            result = self._remove_stopwords(result)

        return result

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Apply preprocessing to a batch of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            List of preprocessed text strings.
        """
        return [self.preprocess(text) for text in texts]

    def _expand_contractions(self, text: str) -> str:
        """Expand contractions like "that's" -> "that is".

        Args:
            text: Input text.

        Returns:
            Text with contractions expanded.
        """
        return str(contractions.fix(text))

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text.

        Args:
            text: Input text.

        Returns:
            Text with punctuation removed.
        """
        return re.sub(self._punctuation_pattern, " ", text)

    def _remove_stopwords(self, text: str) -> str:
        """Remove English stopwords from text.

        Args:
            text: Input text.

        Returns:
            Text with stopwords removed.
        """
        if not self._stopwords:
            return text

        words = text.split()
        filtered = [w for w in words if w not in self._stopwords]
        return " ".join(filtered)
