"""
Model factory for centralized model instantiation.

Factory Pattern: Centralizes model creation so that different strategies
can be instantiated by name without knowing their implementation details.
"""

from typing import Dict, List, Type

from src.config import ExperimentConfig
from src.models.strategies import (
    BowLogisticStrategy,
    ModelStrategy,
    TfidfLogisticStrategy,
)


class ModelFactory:
    """Factory for creating model strategies.

    Factory Pattern: Centralizes model instantiation. Usage:
        strategy = ModelFactory.create("bow", config)

    To add a new model:
    1. Create strategy class in strategies.py
    2. Register in _strategies dictionary below
    3. Use via: ModelFactory.create("new_model", config)

    Example:
        >>> config = ExperimentConfig.for_bow_baseline()
        >>> strategy = ModelFactory.create("bow", config)
        >>> strategy.train(train_texts, train_labels)
    """

    # Registry of available strategies
    _strategies: Dict[str, Type[ModelStrategy]] = {
        "bow": BowLogisticStrategy,
        "tfidf": TfidfLogisticStrategy,
    }

    @classmethod
    def create(cls, model_type: str, config: ExperimentConfig) -> ModelStrategy:
        """Create a model strategy by name.

        Args:
            model_type: Strategy identifier ('bow', 'tfidf').
            config: ExperimentConfig containing model and preprocessing settings.

        Returns:
            Configured ModelStrategy instance.

        Raises:
            ValueError: If model_type is not registered.
        """
        if model_type not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. " f"Available types: {available}"
            )

        strategy_class = cls._strategies[model_type]
        return strategy_class(config.model, config.preprocessing)

    @classmethod
    def register(cls, name: str, strategy_class: Type[ModelStrategy]) -> None:
        """Register a new strategy type.

        Args:
            name: Strategy identifier to use with create().
            strategy_class: Class implementing ModelStrategy interface.

        Example:
            >>> ModelFactory.register("svm", SvmStrategy)
            >>> strategy = ModelFactory.create("svm", config)
        """
        cls._strategies[name] = strategy_class

    @classmethod
    def available_models(cls) -> List[str]:
        """Get list of available model types.

        Returns:
            List of registered model type names.
        """
        return list(cls._strategies.keys())
