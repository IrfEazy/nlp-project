"""
Text preprocessing utilities.

Provides text cleaning operations in the original notebook's order: lowercasing,
punctuation replaced by spaces, contraction expansion, word tokenization and
stopword removal. The LSTM strategy uses it.
"""

import re
import string
from typing import List, Optional

import contractions
import nltk
from nltk.corpus import stopwords

from src.config import PreprocessingConfig

# The original notebook's punctuation class, unescaped on purpose: the ``\]`` it
# contains escapes the bracket, so the backslash itself is *not* matched.
# Many Yelp texts hold a literal backslash-n that must survive.
PUNCTUATION_PATTERN = "[" + string.punctuation + "]"


class TextPreprocessor:
    """Cleans and preprocesses text for NLP tasks.

    Single Responsibility: Only clean text, does NOT vectorize.

    The preprocessing logic differs by model type:
    - Traditional ML (BoW, TF-IDF): remove_stopwords=True to reduce noise
    - Transformer-style models (not implemented here): remove_stopwords=False
      because words like "not", "but" carry sentiment signal

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
        self._punctuation_pattern = PUNCTUATION_PATTERN
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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """NLTK word tokenization (downloads the punkt data if necessary)."""
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return nltk.word_tokenize(text)

    def preprocess(self, text: str) -> str:
        """Apply all configured preprocessing steps to a single text.

        Order (notebook ``remove_noise``): lowercase, punctuation -> space,
        contractions, ``nltk.word_tokenize``, stopword removal, join by spaces.

        Args:
            text: Raw text string.

        Returns:
            Preprocessed text string.
        """
        result = text

        if self.config.lowercase:
            result = result.lower()

        result = self._remove_punctuation(result)

        if self.config.expand_contractions:
            result = self._expand_contractions(result)

        tokens = self._tokenize(result)
        if self.config.remove_stopwords and self._stopwords:
            tokens = [w for w in tokens if w not in self._stopwords]
        return " ".join(tokens)

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
