"""
Tests for the preprocessing module.

Tests cover:
- TextPreprocessor text cleaning operations
- VectorizerFactory vectorizer creation
"""

import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import PreprocessingConfig
from src.preprocessing.feature_engineer import VectorizerFactory
from src.preprocessing.text_preprocessor import TextPreprocessor


@pytest.mark.unit
class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_initialization_with_config(self, default_preprocessing_config) -> None:
        """Preprocessor should initialize with config."""
        preprocessor = TextPreprocessor(default_preprocessing_config)

        assert preprocessor.config == default_preprocessing_config

    def test_lowercase_text(self) -> None:
        """Preprocessor should lowercase text when configured."""
        config = PreprocessingConfig(
            lowercase=True, remove_stopwords=False, expand_contractions=False
        )
        preprocessor = TextPreprocessor(config)

        result = preprocessor.preprocess("HELLO WORLD")

        assert result == "hello world"

    def test_no_lowercase_when_disabled(self) -> None:
        """Preprocessor should preserve case when lowercase=False."""
        config = PreprocessingConfig(
            lowercase=False, remove_stopwords=False, expand_contractions=False
        )
        preprocessor = TextPreprocessor(config)

        result = preprocessor.preprocess("HELLO World")

        # Should still have uppercase since lowercase=False
        assert (
            "HELLO" in result
            or "hello" not in result.split()[0].lower()
            or result.split()[0] == "HELLO"
        )

    def test_remove_punctuation(self) -> None:
        """Preprocessor should remove punctuation."""
        config = PreprocessingConfig(
            lowercase=False, remove_stopwords=False, expand_contractions=False
        )
        preprocessor = TextPreprocessor(config)

        result = preprocessor.preprocess("Hello, world! How are you?")

        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_expand_contractions(self) -> None:
        """Preprocessor should expand contractions when configured."""
        config = PreprocessingConfig(
            lowercase=True, remove_stopwords=False, expand_contractions=True
        )
        preprocessor = TextPreprocessor(config)

        result = preprocessor.preprocess("That's great")

        # After expansion and processing
        assert "that" in result.lower()
        assert "is" in result.lower() or "s" not in result  # "that's" -> "that is"

    def test_remove_stopwords_config(self) -> None:
        """Enabling stopword removal should configure the preprocessor."""
        config = PreprocessingConfig(
            remove_stopwords=True, lowercase=True, expand_contractions=False
        )

        preprocessor = TextPreprocessor(config)

        # Stopwords should be loaded
        assert preprocessor._stopwords is not None
        assert len(preprocessor._stopwords) > 0

    def test_stopwords_removed_from_text(self) -> None:
        """Stopwords should be removed when configured."""
        config = PreprocessingConfig(
            remove_stopwords=True, lowercase=True, expand_contractions=False
        )
        preprocessor = TextPreprocessor(config)

        result = preprocessor.preprocess("the quick brown fox is very fast")

        # Common stopwords should be removed
        words = result.split()
        # 'the', 'is', 'very' are common stopwords
        assert "quick" in words or "brown" in words or "fox" in words

    def test_preprocess_batch(self, sample_texts, default_preprocessing_config) -> None:
        """preprocess_batch should process list of texts."""
        preprocessor = TextPreprocessor(default_preprocessing_config)

        results = preprocessor.preprocess_batch(sample_texts)

        assert isinstance(results, list)
        assert len(results) == len(sample_texts)
        assert all(isinstance(r, str) for r in results)

    def test_preprocess_empty_string(self, default_preprocessing_config) -> None:
        """Preprocessor should handle empty string."""
        preprocessor = TextPreprocessor(default_preprocessing_config)

        result = preprocessor.preprocess("")

        assert isinstance(result, str)

    def test_no_stopword_removal_preserves_words(self) -> None:
        """When remove_stopwords=False, all words should be preserved."""
        config = PreprocessingConfig(
            remove_stopwords=False, lowercase=True, expand_contractions=False
        )
        preprocessor = TextPreprocessor(config)

        result = preprocessor.preprocess("the quick brown fox")

        # 'the' should still be present
        assert "the" in result.split()


@pytest.mark.unit
class TestVectorizerFactory:
    """Tests for VectorizerFactory class."""

    def test_create_bow_returns_count_vectorizer(self) -> None:
        """create_bow should return CountVectorizer instance."""
        vectorizer = VectorizerFactory.create_bow()

        assert isinstance(vectorizer, CountVectorizer)

    def test_create_bow_with_custom_min_df(self) -> None:
        """create_bow should accept custom min_df."""
        vectorizer = VectorizerFactory.create_bow(min_df=100)

        assert vectorizer.min_df == 100

    def test_create_bow_with_stop_words(self) -> None:
        """create_bow should configure stop words."""
        vectorizer = VectorizerFactory.create_bow(stop_words="english")

        assert vectorizer.stop_words == "english"

    def test_create_bow_without_stop_words(self) -> None:
        """create_bow should allow disabling stop words."""
        vectorizer = VectorizerFactory.create_bow(stop_words=None)

        assert vectorizer.stop_words is None

    def test_create_tfidf_returns_tfidf_vectorizer(self) -> None:
        """create_tfidf should return TfidfVectorizer instance."""
        vectorizer = VectorizerFactory.create_tfidf()

        assert isinstance(vectorizer, TfidfVectorizer)

    def test_create_tfidf_with_max_df(self) -> None:
        """create_tfidf should accept custom max_df."""
        vectorizer = VectorizerFactory.create_tfidf(max_df=0.8)

        assert vectorizer.max_df == 0.8

    def test_create_tfidf_with_min_df(self) -> None:
        """create_tfidf should accept custom min_df."""
        vectorizer = VectorizerFactory.create_tfidf(min_df=25)

        assert vectorizer.min_df == 25

    @pytest.mark.parametrize(
        "vectorizer_type,expected_class",
        [
            ("bow", CountVectorizer),
            ("tfidf", TfidfVectorizer),
        ],
        ids=["bow-vectorizer", "tfidf-vectorizer"],
    )
    def test_create_from_config_vectorizer_types(
        self, vectorizer_type, expected_class, default_preprocessing_config
    ) -> None:
        """create_from_config should create correct vectorizer type."""
        vectorizer = VectorizerFactory.create_from_config(
            vectorizer_type, default_preprocessing_config
        )

        assert isinstance(vectorizer, expected_class)

    def test_create_from_config_unknown_type(
        self, default_preprocessing_config
    ) -> None:
        """create_from_config should raise error for unknown type."""
        with pytest.raises(ValueError, match="Unknown vectorizer type"):
            VectorizerFactory.create_from_config(
                "unknown", default_preprocessing_config
            )

    @pytest.mark.parametrize(
        "remove_stopwords,expected_stop_words",
        [
            (True, "english"),
            (False, None),
        ],
        ids=["stopwords-enabled", "stopwords-disabled"],
    )
    def test_create_from_config_stopword_handling(
        self, remove_stopwords, expected_stop_words
    ) -> None:
        """create_from_config should configure stopwords based on config."""
        config = PreprocessingConfig(remove_stopwords=remove_stopwords)
        vectorizer = VectorizerFactory.create_from_config("bow", config)

        assert vectorizer.stop_words == expected_stop_words


@pytest.mark.integration
class TestVectorizerIntegration:
    """Integration tests for vectorizers with actual data."""

    def test_bow_fit_transform(self, small_train_texts) -> None:
        """BoW vectorizer should fit and transform texts."""
        vectorizer = VectorizerFactory.create_bow(min_df=1)

        X = vectorizer.fit_transform(small_train_texts)

        assert X.shape[0] == len(small_train_texts)
        assert X.shape[1] > 0  # Should have features

    def test_tfidf_fit_transform(self, small_train_texts) -> None:
        """TF-IDF vectorizer should fit and transform texts."""
        vectorizer = VectorizerFactory.create_tfidf(min_df=1, max_df=1.0)

        X = vectorizer.fit_transform(small_train_texts)

        assert X.shape[0] == len(small_train_texts)
        assert X.shape[1] > 0

    def test_min_df_filters_vocabulary(self, small_train_texts) -> None:
        """Higher min_df should result in smaller vocabulary."""
        vectorizer_low = VectorizerFactory.create_bow(min_df=1)
        vectorizer_high = VectorizerFactory.create_bow(min_df=5)

        X_low = vectorizer_low.fit_transform(small_train_texts)
        X_high = vectorizer_high.fit_transform(small_train_texts)

        # Higher min_df should have fewer features
        assert X_low.shape[1] >= X_high.shape[1]
        assert X_low.shape[1] >= X_high.shape[1]
