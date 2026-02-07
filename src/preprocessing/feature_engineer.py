"""
Feature engineering utilities.

Provides factory methods for creating vectorizers (BoW, TF-IDF) with
consistent configuration. Never instantiate vectorizers directly -
use VectorizerFactory for consistency.
"""

from typing import Optional

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class VectorizerFactory:
    """Factory for creating text vectorizers with consistent settings.

    Factory Pattern: Centralizes vectorizer creation with predefined
    configurations. Use factory methods instead of instantiating
    CountVectorizer/TfidfVectorizer directly.

    Example:
        >>> vectorizer = VectorizerFactory.create_bow(min_df=50)
        >>> X = vectorizer.fit_transform(texts)
    """

    @staticmethod
    def create_bow(
        min_df: int = 50,
        max_df: float = 1.0,
        stop_words: Optional[str] = "english",
        lowercase: bool = True,
    ) -> CountVectorizer:
        """Create a Bag-of-Words vectorizer.

        Args:
            min_df: Minimum document frequency. Words in fewer documents are excluded.
                    Default 50 is crucial for preventing 100K+ feature spaces.
            max_df: Maximum document frequency (fraction). Default 1.0 (no filtering).
            stop_words: Language for stopwords removal or None to disable.
            lowercase: Whether to convert text to lowercase.

        Returns:
            Configured CountVectorizer instance.
        """
        return CountVectorizer(
            min_df=min_df, max_df=max_df, stop_words=stop_words, lowercase=lowercase
        )

    @staticmethod
    def create_tfidf(
        min_df: int = 50,
        max_df: float = 0.5,
        stop_words: Optional[str] = "english",
        lowercase: bool = True,
    ) -> TfidfVectorizer:
        """Create a TF-IDF vectorizer.

        Args:
            min_df: Minimum document frequency. Default 50.
            max_df: Maximum document frequency (fraction). Default 0.5.
            stop_words: Language for stopwords removal or None to disable.
            lowercase: Whether to convert text to lowercase.

        Returns:
            Configured TfidfVectorizer instance.
        """
        return TfidfVectorizer(
            min_df=min_df, max_df=max_df, stop_words=stop_words, lowercase=lowercase
        )

    @staticmethod
    def create_from_config(
        model_type: str, config
    ) -> "CountVectorizer | TfidfVectorizer":
        """Create a vectorizer based on model type and preprocessing config.

        Args:
            model_type: Either 'bow' or 'tfidf'.
            config: PreprocessingConfig object.

        Returns:
            Configured vectorizer instance.

        Raises:
            ValueError: If model_type is not 'bow' or 'tfidf'.
        """
        stop_words = "english" if config.remove_stopwords else None

        if model_type == "bow":
            return VectorizerFactory.create_bow(
                min_df=config.min_df,
                max_df=config.max_df,
                stop_words=stop_words,
                lowercase=config.lowercase,
            )
        elif model_type == "tfidf":
            return VectorizerFactory.create_tfidf(
                min_df=config.min_df,
                max_df=config.max_df,
                stop_words=stop_words,
                lowercase=config.lowercase,
            )
        else:
            raise ValueError(
                f"Unknown vectorizer type: {model_type}. Use 'bow' or 'tfidf'."
            )
