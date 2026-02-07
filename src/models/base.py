"""
Base model abstractions.

Defines the interface that all models must implement, enabling
consistent interaction regardless of the underlying implementation.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseModel(ABC):
    """Abstract base class for all models.

    Defines the interface that all models must implement. This enables
    the Adapter pattern where sklearn models can be wrapped to implement
    this interface.

    Interface Segregation Principle: Keep the interface minimal with
    only the methods that all models need.
    """

    @abstractmethod
    def fit(self, X, y) -> "BaseModel":
        """Train the model on the provided data.

        Args:
            X: Feature matrix (can be sparse or dense).
            y: Target labels.

        Returns:
            self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X) -> NDArray[np.int_]:
        """Make predictions on new data.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted labels.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get a human-readable name for the model.

        Returns:
            Model name string.
        """
        pass
