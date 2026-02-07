"""
Model adapters for wrapping external models.

Adapter Pattern: Wraps sklearn (or other) models to implement the
BaseModel interface, enabling uniform usage regardless of the
underlying implementation.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.models.base import BaseModel


class SklearnModelAdapter(BaseModel):
    """Adapter wrapping sklearn models to implement BaseModel interface.

    Adapter Pattern: Allows sklearn models (LogisticRegression, SVC, etc.)
    to be used interchangeably with other models implementing BaseModel.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> sklearn_model = LogisticRegression()
        >>> model = SklearnModelAdapter(sklearn_model, "Logistic Regression")
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, sklearn_model: Any, name: str = "Sklearn Model") -> None:
        """Initialize the adapter with a sklearn model.

        Args:
            sklearn_model: Any sklearn estimator with fit/predict methods.
            name: Human-readable name for the model.
        """
        self._model = sklearn_model
        self._name = name

    def fit(self, X, y) -> "SklearnModelAdapter":
        """Train the wrapped sklearn model.

        Args:
            X: Feature matrix (can be sparse or dense).
            y: Target labels.

        Returns:
            self for method chaining.
        """
        self._model.fit(X, y)
        return self

    def predict(self, X) -> NDArray[np.int_]:
        """Make predictions using the wrapped model.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted labels.
        """
        return self._model.predict(X)

    def get_name(self) -> str:
        """Get the model name.

        Returns:
            Human-readable model name.
        """
        return self._name

    @property
    def wrapped_model(self) -> Any:
        """Access the underlying sklearn model.

        Useful for accessing model-specific attributes like coefficients.

        Returns:
            The wrapped sklearn model instance.
        """
        return self._model
