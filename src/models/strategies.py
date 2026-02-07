"""
Model strategies implementing the Strategy Pattern.

Encapsulates entire model pipelines (vectorization + classification) as
interchangeable strategies. Each strategy handles its own feature extraction
and model training.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from src.config import ModelConfig, PreprocessingConfig
from src.preprocessing.feature_engineer import VectorizerFactory


class ModelStrategy(ABC):
    """Abstract base class for model strategies.

    Strategy Pattern: Encapsulates the algorithm (vectorization + model)
    so that strategies can be swapped without changing the pipeline code.

    Each strategy is responsible for:
    1. Vectorizing raw text (using appropriate vectorizer)
    2. Training the classification model
    3. Making predictions on new text
    """

    @abstractmethod
    def train(self, train_texts: List[str], train_labels: List[int]) -> None:
        """Train the model on raw text data.

        Args:
            train_texts: List of training document strings.
            train_labels: List of integer labels (0-4).
        """
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> NDArray[np.int_]:
        """Make predictions on raw text data.

        Args:
            texts: List of document strings to classify.

        Returns:
            Array of predicted integer labels.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get a human-readable name for this strategy.

        Returns:
            Strategy name string.
        """
        pass


class BowLogisticStrategy(ModelStrategy):
    """Bag-of-Words with Logistic Regression strategy.

    Baseline model using CountVectorizer for feature extraction
    and Logistic Regression for classification.

    Example:
        >>> strategy = BowLogisticStrategy(model_config, preproc_config)
        >>> strategy.train(train_texts, train_labels)
        >>> predictions = strategy.predict(test_texts)
    """

    def __init__(
        self, model_config: ModelConfig, preprocessing_config: PreprocessingConfig
    ) -> None:
        """Initialize the BoW + Logistic Regression strategy.

        Args:
            model_config: Model configuration.
            preprocessing_config: Preprocessing configuration.
        """
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.vectorizer = None
        self.model = None

    def train(self, train_texts: List[str], train_labels: List[int]) -> None:
        """Train BoW vectorizer and Logistic Regression model.

        Args:
            train_texts: List of training document strings.
            train_labels: List of integer labels (0-4).
        """
        # Create vectorizer based on config
        stop_words = "english" if self.preprocessing_config.remove_stopwords else None
        self.vectorizer = VectorizerFactory.create_bow(
            min_df=self.preprocessing_config.min_df,
            stop_words=stop_words,
            lowercase=self.preprocessing_config.lowercase,
        )

        # Vectorize training data
        X_train = self.vectorizer.fit_transform(train_texts)

        # Train logistic regression
        self.model = LogisticRegression(
            max_iter=self.model_config.max_iter,
            random_state=self.model_config.random_seed,
        )
        self.model.fit(X_train, train_labels)

    def predict(self, texts: List[str]) -> NDArray[np.int_]:
        """Predict labels for new texts.

        Args:
            texts: List of document strings to classify.

        Returns:
            Array of predicted integer labels.

        Raises:
            RuntimeError: If called before train().
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError(
                "Strategy must be trained before prediction. Call train() first."
            )

        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def get_name(self) -> str:
        """Get strategy name."""
        return "Bag-of-Words + Logistic Regression"

    def get_vocabulary_size(self) -> int:
        """Get the size of the learned vocabulary.

        Returns:
            Number of features in the vocabulary.

        Raises:
            RuntimeError: If called before train().
        """
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted. Call train() first.")
        return len(self.vectorizer.get_feature_names_out())

    def get_top_features(self, n: int = 10) -> Dict[int, List[str]]:
        """Get top features for each class by coefficient weight.

        Args:
            n: Number of top features to return per class.

        Returns:
            Dictionary mapping class label to list of top feature names.
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Strategy must be trained first.")

        vocab = self.vectorizer.get_feature_names_out()
        result = {}

        for label in range(self.model_config.num_classes):
            top_indices = np.argsort(self.model.coef_[label])[-n:][::-1]
            result[label] = [vocab[i] for i in top_indices]

        return result


class TfidfLogisticStrategy(ModelStrategy):
    """TF-IDF with Logistic Regression strategy.

    Uses TfidfVectorizer for feature extraction and Logistic Regression
    for classification. Generally provides slight improvement over BoW.

    Example:
        >>> strategy = TfidfLogisticStrategy(model_config, preproc_config)
        >>> strategy.train(train_texts, train_labels)
        >>> predictions = strategy.predict(test_texts)
    """

    def __init__(
        self, model_config: ModelConfig, preprocessing_config: PreprocessingConfig
    ) -> None:
        """Initialize the TF-IDF + Logistic Regression strategy.

        Args:
            model_config: Model configuration.
            preprocessing_config: Preprocessing configuration.
        """
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.vectorizer = None
        self.model = None

    def train(self, train_texts: List[str], train_labels: List[int]) -> None:
        """Train TF-IDF vectorizer and Logistic Regression model.

        Args:
            train_texts: List of training document strings.
            train_labels: List of integer labels (0-4).
        """
        # Create vectorizer based on config
        stop_words = "english" if self.preprocessing_config.remove_stopwords else None
        self.vectorizer = VectorizerFactory.create_tfidf(
            min_df=self.preprocessing_config.min_df,
            max_df=self.preprocessing_config.max_df,
            stop_words=stop_words,
            lowercase=self.preprocessing_config.lowercase,
        )

        # Vectorize training data
        X_train = self.vectorizer.fit_transform(train_texts)

        # Train logistic regression
        self.model = LogisticRegression(
            max_iter=self.model_config.max_iter,
            random_state=self.model_config.random_seed,
        )
        self.model.fit(X_train, train_labels)

    def predict(self, texts: List[str]) -> NDArray[np.int_]:
        """Predict labels for new texts.

        Args:
            texts: List of document strings to classify.

        Returns:
            Array of predicted integer labels.

        Raises:
            RuntimeError: If called before train().
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError(
                "Strategy must be trained before prediction. Call train() first."
            )

        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def get_name(self) -> str:
        """Get strategy name."""
        return "TF-IDF + Logistic Regression"

    def get_vocabulary_size(self) -> int:
        """Get the size of the learned vocabulary.

        Returns:
            Number of features in the vocabulary.
        """
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted. Call train() first.")
        return len(self.vectorizer.get_feature_names_out())

    def get_top_features(self, n: int = 10) -> Dict[int, List[str]]:
        """Get top features for each class by coefficient weight.

        Args:
            n: Number of top features to return per class.

        Returns:
            Dictionary mapping class label to list of top feature names.
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Strategy must be trained first.")

        vocab = self.vectorizer.get_feature_names_out()
        result = {}

        for label in range(self.model_config.num_classes):
            top_indices = np.argsort(self.model.coef_[label])[-n:][::-1]
            result[label] = [vocab[i] for i in top_indices]

        return result
