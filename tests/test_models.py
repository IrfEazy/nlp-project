"""
Tests for the models module.

Tests cover:
- Model strategies (BoW, TF-IDF)
- Model factory
- Sklearn model adapter
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from src.config import ExperimentConfig, ModelConfig, PreprocessingConfig
from src.models.adapters import SklearnModelAdapter
from src.models.base import BaseModel
from src.models.factory import ModelFactory
from src.preprocessing.feature_engineer import delete_punctuation
from src.models.strategies import (
    BowLogisticStrategy,
    ModelStrategy,
    TfidfLogisticStrategy,
)


@pytest.mark.unit
class TestBowLogisticStrategy:
    """Tests for BowLogisticStrategy."""

    def test_implements_model_strategy(
        self, default_model_config, default_preprocessing_config
    ) -> None:
        """Strategy should implement ModelStrategy interface."""
        strategy = BowLogisticStrategy(
            default_model_config, default_preprocessing_config
        )
        assert isinstance(strategy, ModelStrategy)

    def test_train_creates_vectorizer_and_model(
        self,
        small_train_texts,
        small_train_labels,
        default_model_config,
        default_preprocessing_config,
    ) -> None:
        """After training, vectorizer and model should be set."""
        strategy = BowLogisticStrategy(
            default_model_config, default_preprocessing_config
        )

        assert strategy.vectorizer is None
        assert strategy.model is None

        strategy.train(small_train_texts, small_train_labels)

        assert strategy.vectorizer is not None
        assert strategy.model is not None

    def test_predict_returns_array(
        self,
        small_train_texts,
        small_train_labels,
        small_test_texts,
        default_model_config,
        default_preprocessing_config,
    ) -> None:
        """Predict should return numpy array of labels."""
        strategy = BowLogisticStrategy(
            default_model_config, default_preprocessing_config
        )
        strategy.train(small_train_texts, small_train_labels)

        predictions = strategy.predict(small_test_texts)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(small_test_texts)
        assert all(0 <= p <= 4 for p in predictions)

    def test_predict_before_train_raises_error(
        self, small_test_texts, default_model_config, default_preprocessing_config
    ) -> None:
        """Predict before train should raise RuntimeError."""
        strategy = BowLogisticStrategy(
            default_model_config, default_preprocessing_config
        )

        with pytest.raises(RuntimeError, match="must be trained"):
            strategy.predict(small_test_texts)

    def test_get_name_returns_string(
        self, default_model_config, default_preprocessing_config
    ) -> None:
        """get_name should return descriptive string."""
        strategy = BowLogisticStrategy(
            default_model_config, default_preprocessing_config
        )
        name = strategy.get_name()

        assert isinstance(name, str)
        assert "Bag-of-Words" in name
        assert "Logistic" in name

    def test_get_vocabulary_size(
        self,
        small_train_texts,
        small_train_labels,
        default_model_config,
        default_preprocessing_config,
    ) -> None:
        """Vocabulary size should be positive after training."""
        strategy = BowLogisticStrategy(
            default_model_config, default_preprocessing_config
        )
        strategy.train(small_train_texts, small_train_labels)

        vocab_size = strategy.get_vocabulary_size()

        assert vocab_size > 0

    def test_get_top_features(
        self,
        small_train_texts,
        small_train_labels,
        default_model_config,
        default_preprocessing_config,
    ) -> None:
        """get_top_features should return dict with features per class."""
        strategy = BowLogisticStrategy(
            default_model_config, default_preprocessing_config
        )
        strategy.train(small_train_texts, small_train_labels)

        top_features = strategy.get_top_features(n=3)

        assert isinstance(top_features, dict)
        assert len(top_features) == 5  # 5 classes


@pytest.mark.unit
class TestTfidfLogisticStrategy:
    """Tests for TfidfLogisticStrategy."""

    def test_implements_model_strategy(
        self, default_model_config, default_preprocessing_config
    ) -> None:
        """Strategy should implement ModelStrategy interface."""
        strategy = TfidfLogisticStrategy(
            default_model_config, default_preprocessing_config
        )
        assert isinstance(strategy, ModelStrategy)

    def test_train_and_predict(
        self,
        small_train_texts,
        small_train_labels,
        small_test_texts,
        default_model_config,
        default_preprocessing_config,
    ) -> None:
        """Strategy should train and predict successfully."""
        # Use TF-IDF specific config
        preprocessing_config = PreprocessingConfig(
            min_df=2, max_df=0.95, remove_stopwords=True
        )

        strategy = TfidfLogisticStrategy(default_model_config, preprocessing_config)
        strategy.train(small_train_texts, small_train_labels)
        predictions = strategy.predict(small_test_texts)

        assert len(predictions) == len(small_test_texts)

    def test_get_name_returns_tfidf(
        self, default_model_config, default_preprocessing_config
    ) -> None:
        """get_name should mention TF-IDF."""
        strategy = TfidfLogisticStrategy(
            default_model_config, default_preprocessing_config
        )
        assert "TF-IDF" in strategy.get_name()


@pytest.mark.unit
class TestModelFactory:
    """Tests for ModelFactory."""

    def test_create_bow_strategy(self, default_experiment_config) -> None:
        """Factory should create BowLogisticStrategy for 'bow'."""
        strategy = ModelFactory.create("bow", default_experiment_config)
        assert isinstance(strategy, BowLogisticStrategy)

    def test_create_tfidf_strategy(self, default_experiment_config) -> None:
        """Factory should create TfidfLogisticStrategy for 'tfidf'."""
        default_experiment_config.model.model_type = "tfidf"
        strategy = ModelFactory.create("tfidf", default_experiment_config)
        assert isinstance(strategy, TfidfLogisticStrategy)

    def test_create_unknown_raises_error(self, default_experiment_config) -> None:
        """Unknown model type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create("unknown_model", default_experiment_config)

    def test_available_models(self) -> None:
        """available_models should return list of registered types."""
        models = ModelFactory.available_models()

        assert isinstance(models, list)
        assert "bow" in models
        assert "tfidf" in models

    def test_register_custom_strategy(self, default_experiment_config) -> None:
        """Should be able to register and use custom strategies."""

        # Create a mock strategy class
        class MockStrategy(ModelStrategy):
            def __init__(self, model_config, preprocessing_config) -> None:
                pass

            def train(self, texts, labels) -> None:
                pass

            def predict(self, texts) -> NDArray[np.int_]:
                return np.array([0] * len(texts))

            def get_name(self) -> str:
                return "Mock"

        # Register it
        ModelFactory.register("mock", MockStrategy)

        # Create using factory
        strategy = ModelFactory.create("mock", default_experiment_config)
        assert isinstance(strategy, MockStrategy)

        # Clean up
        del ModelFactory._strategies["mock"]


@pytest.mark.unit
class TestSklearnModelAdapter:
    """Tests for SklearnModelAdapter."""

    def test_implements_base_model(self) -> None:
        """Adapter should implement BaseModel interface."""
        mock_model = MagicMock()
        adapter = SklearnModelAdapter(mock_model, "Test Model")
        assert isinstance(adapter, BaseModel)

    def test_fit_delegates_to_wrapped_model(self) -> None:
        """fit should call wrapped model's fit method."""
        mock_model = MagicMock()
        adapter = SklearnModelAdapter(mock_model)

        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        result = adapter.fit(X, y)

        mock_model.fit.assert_called_once_with(X, y)
        assert result is adapter  # Returns self for chaining

    def test_predict_delegates_to_wrapped_model(self) -> None:
        """predict should call wrapped model's predict method."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 2])
        adapter = SklearnModelAdapter(mock_model)

        X = np.array([[1, 2], [3, 4], [5, 6]])

        predictions = adapter.predict(X)

        mock_model.predict.assert_called_once_with(X)
        assert np.array_equal(predictions, np.array([0, 1, 2]))

    def test_get_name(self) -> None:
        """get_name should return the configured name."""
        mock_model = MagicMock()
        adapter = SklearnModelAdapter(mock_model, "Custom Name")

        assert adapter.get_name() == "Custom Name"

    def test_wrapped_model_property(self) -> None:
        """wrapped_model should provide access to underlying model."""
        mock_model = MagicMock()
        adapter = SklearnModelAdapter(mock_model)

        assert adapter.wrapped_model is mock_model


@pytest.mark.integration
class TestStrategyIntegration:
    """Integration tests for strategies with real sklearn models."""

    @pytest.mark.parametrize("strategy_type", ["bow", "tfidf"])
    def test_full_train_predict_cycle(
        self,
        strategy_type,
        small_train_texts,
        small_train_labels,
        small_test_texts,
        default_experiment_config,
    ) -> None:
        """Full training and prediction cycle should work for both strategies."""
        default_experiment_config.model.model_type = strategy_type
        default_experiment_config.preprocessing.min_df = 2

        strategy = ModelFactory.create(strategy_type, default_experiment_config)
        strategy.train(small_train_texts, small_train_labels)
        predictions = strategy.predict(small_test_texts)

        # Verify predictions are valid
        assert len(predictions) == len(small_test_texts)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
        assert all(0 <= p <= 4 for p in predictions)

        # Verify predictions are valid
        assert len(predictions) == len(small_test_texts)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
        assert all(0 <= p <= 4 for p in predictions)
        assert all(0 <= p <= 4 for p in predictions)


@pytest.mark.unit
class TestLogisticOptions:
    """class_weight, punctuation deletion and top features by fitted classes."""

    TEXTS = ["good, great!", "bad; awful.", "good great"] * 30
    LABELS = [1, 0, 1] * 30

    def make(self, cls=BowLogisticStrategy, **model_kwargs):
        prep = PreprocessingConfig(min_df=2, max_df=1.0, strip_punctuation=model_kwargs.pop("strip", False))
        return cls(ModelConfig(**model_kwargs), prep)

    def test_class_weight_passed_to_logistic_regression(self) -> None:
        strategy = self.make(class_weight="balanced")
        strategy.train(self.TEXTS, self.LABELS)
        assert strategy.model.class_weight == "balanced"

    def test_default_max_iter_and_no_class_weight(self) -> None:
        strategy = self.make()
        strategy.train(self.TEXTS, self.LABELS)
        assert strategy.model.max_iter == 100
        assert strategy.model.class_weight is None

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_custom_max_iter_respected(self) -> None:
        strategy = self.make(max_iter=7)
        strategy.train(self.TEXTS, self.LABELS)
        assert strategy.model.max_iter == 7

    def test_punctuation_deleted_before_vectorizer(self) -> None:
        strategy = self.make(strip=True)
        strategy.train(["don't stop", "dont stop"] * 20, [0, 1] * 20)
        assert "dont" in strategy.vectorizer.vocabulary_
        assert strategy.predict(["don't"]).shape == (1,)

    def test_punctuation_kept_by_default(self) -> None:
        strategy = self.make()
        strategy.train(["don't stop", "dont stop"] * 20, [0, 1] * 20)
        assert "don" in strategy.vectorizer.vocabulary_

    def test_delete_punctuation_helper(self) -> None:
        assert delete_punctuation(["a,b", "don't!"]) == ["ab", "dont"]

    def test_delete_punctuation_keeps_backslash_like_the_notebook(self) -> None:
        # '[' + string.punctuation + ']' escapes the bracket and misses "\\"
        assert delete_punctuation(["a\\nb,c"]) == ["a\\nbc"]

    @pytest.mark.parametrize("cls", [BowLogisticStrategy, TfidfLogisticStrategy])
    def test_top_features_binary_and_three_class(self, cls) -> None:
        strategy = self.make(cls)
        strategy.train(self.TEXTS, self.LABELS)
        top = strategy.get_top_features(n=1)
        assert set(top) == {0, 1}
        assert all(len(features) == 1 for features in top.values())
        assert top[0] != top[1]

        strategy = self.make(cls)
        strategy.train(["a1 x", "b1 y", "c1 z"] * 20, [0, 1, 2] * 20)
        assert set(strategy.get_top_features(n=1)) == {0, 1, 2}


class _LazyStub:
    """Records the arguments the factory gives to lazily registered strategies."""

    def __init__(self, model_config, preprocessing_config, n_classes=None) -> None:
        self.n_classes = n_classes


@pytest.mark.unit
class TestFactoryLazy:
    """Lazy registration does not import until create()."""

    def test_lazy_registration(self) -> None:
        try:
            ModelFactory.register_lazy("lazy-test", "tests.test_models:_LazyStub")
            assert "lazy-test" in ModelFactory.available_models()
            strategy = ModelFactory.create("lazy-test", ExperimentConfig())
            assert isinstance(strategy, _LazyStub)
            assert strategy.n_classes == 5  # default task stars5
            three = ExperimentConfig()
            three.data.task = "stars3"
            assert ModelFactory.create("lazy-test", three).n_classes == 3
        finally:
            ModelFactory._lazy.pop("lazy-test", None)

    def test_lazy_import_error_propagates(self) -> None:
        try:
            ModelFactory.register_lazy("lazy-missing", "no_such_module_xyz:Cls")
            with pytest.raises(ImportError):
                ModelFactory.create("lazy-missing", ExperimentConfig())
        finally:
            ModelFactory._lazy.pop("lazy-missing", None)
