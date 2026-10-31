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
from src.preprocessing.feature_engineer import VectorizerFactory, delete_punctuation


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


class _LogisticStrategy(ModelStrategy):
    """Shared vectorizer + LogisticRegression pipeline.

    Subclasses provide ``_create_vectorizer`` and ``get_name``.
    """

    def __init__(
        self, model_config: ModelConfig, preprocessing_config: PreprocessingConfig
    ) -> None:
        """Initialize the strategy.

        Args:
            model_config: Model configuration.
            preprocessing_config: Preprocessing configuration.
        """
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.vectorizer = None
        self.model = None

    @abstractmethod
    def _create_vectorizer(self):
        """Build the (unfitted) vectorizer through VectorizerFactory."""

    def _stop_words(self):
        return "english" if self.preprocessing_config.remove_stopwords else None

    def _clean(self, texts: List[str]) -> List[str]:
        if self.preprocessing_config.strip_punctuation:
            return delete_punctuation(texts)
        return texts

    def train(self, train_texts: List[str], train_labels: List[int]) -> None:
        """Train the vectorizer and Logistic Regression model.

        Args:
            train_texts: List of training document strings.
            train_labels: List of integer task labels.
        """
        self.vectorizer = self._create_vectorizer()
        X_train = self.vectorizer.fit_transform(self._clean(train_texts))

        self.model = LogisticRegression(
            max_iter=self.model_config.max_iter,
            random_state=self.model_config.random_seed,
            class_weight=self.model_config.class_weight,
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

        return self.model.predict(self.vectorizer.transform(self._clean(texts)))

    def get_vocabulary_size(self) -> int:
        """Get the size of the learned vocabulary.

        Raises:
            RuntimeError: If called before train().
        """
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted. Call train() first.")
        return len(self.vectorizer.get_feature_names_out())

    def get_top_features(self, n: int = 10) -> Dict[int, List[str]]:
        """Get top features for each class by coefficient weight.

        The class count comes from the fitted model. A binary model has one
        coefficient row for the positive class; the other class uses its
        negation.

        Args:
            n: Number of top features to return per class.

        Returns:
            Dictionary mapping class label to list of top feature names.
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Strategy must be trained first.")

        vocab = self.vectorizer.get_feature_names_out()
        coef = self.model.coef_
        if coef.shape[0] == 1:
            coef = np.vstack([-coef[0], coef[0]])

        result = {}
        for row, label in enumerate(self.model.classes_):
            top_indices = np.argsort(coef[row])[-n:][::-1]
            result[int(label)] = [vocab[i] for i in top_indices]

        return result


class BowLogisticStrategy(_LogisticStrategy):
    """Bag-of-Words with Logistic Regression strategy.

    CountVectorizer features (no ``max_df``, as in the original notebook) and
    Logistic Regression.

    Example:
        >>> strategy = BowLogisticStrategy(model_config, preproc_config)
        >>> strategy.train(train_texts, train_labels)
        >>> predictions = strategy.predict(test_texts)
    """

    def _create_vectorizer(self):
        return VectorizerFactory.create_bow(
            min_df=self.preprocessing_config.min_df,
            stop_words=self._stop_words(),
            lowercase=self.preprocessing_config.lowercase,
        )

    def get_name(self) -> str:
        """Get strategy name."""
        return "Bag-of-Words + Logistic Regression"


class TfidfLogisticStrategy(_LogisticStrategy):
    """TF-IDF with Logistic Regression strategy.

    Example:
        >>> strategy = TfidfLogisticStrategy(model_config, preproc_config)
        >>> strategy.train(train_texts, train_labels)
        >>> predictions = strategy.predict(test_texts)
    """

    def _create_vectorizer(self):
        return VectorizerFactory.create_tfidf(
            min_df=self.preprocessing_config.min_df,
            max_df=self.preprocessing_config.max_df,
            stop_words=self._stop_words(),
            lowercase=self.preprocessing_config.lowercase,
        )

    def get_name(self) -> str:
        """Get strategy name."""
        return "TF-IDF + Logistic Regression"
