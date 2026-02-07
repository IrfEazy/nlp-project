"""
Model evaluation utilities.

Provides metrics computation for classification models including
accuracy, precision, recall, F1-score, and confusion matrix.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics.

    Attributes:
        accuracy: Overall accuracy score.
        precision_per_class: Precision for each class.
        recall_per_class: Recall for each class.
        f1_per_class: F1-score for each class.
        macro_precision: Macro-averaged precision.
        macro_recall: Macro-averaged recall.
        macro_f1: Macro-averaged F1-score.
        confusion_matrix: NxN confusion matrix.
        classification_report: Formatted classification report string.
    """

    accuracy: float
    precision_per_class: Dict[int, float] = field(default_factory=dict)
    recall_per_class: Dict[int, float] = field(default_factory=dict)
    f1_per_class: Dict[int, float] = field(default_factory=dict)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""


class ModelEvaluator:
    """Computes evaluation metrics for classification models.

    Single Responsibility: Only compute metrics, does NOT visualize
    (that's ResultsVisualizer's job) or train models.

    Example:
        >>> evaluator = ModelEvaluator(class_names=['0', '1', '2', '3', '4'])
        >>> metrics = evaluator.evaluate(y_true, y_pred)
        >>> print(f"Accuracy: {metrics.accuracy:.3f}")
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize the evaluator.

        Args:
            class_names: Human-readable names for each class.
                        Defaults to ['0', '1', '2', '3', '4'] for Yelp.
        """
        self.class_names = class_names or ["0", "1", "2", "3", "4"]

    def evaluate(self, y_true: List[int], y_pred: np.ndarray) -> EvaluationMetrics:
        """Compute all evaluation metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            EvaluationMetrics object containing all computed metrics.
        """
        # Define all possible labels for consistent metrics
        all_labels = list(range(len(self.class_names)))

        # Basic accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics with explicit labels
        precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average=None, zero_division=0
        )
        # Cast to ndarray (average=None returns arrays, not floats)
        precision = np.asarray(precision_arr)
        recall = np.asarray(recall_arr)
        f1 = np.asarray(f1_arr)

        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average="macro", zero_division=0
        )

        # Confusion matrix with explicit labels
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        # Classification report with explicit labels
        report = classification_report(
            y_true,
            y_pred,
            labels=all_labels,
            target_names=self.class_names,
            zero_division=0,
        )

        # Build per-class dictionaries
        num_classes = len(self.class_names)
        precision_dict = {i: float(precision[i]) for i in range(num_classes)}
        recall_dict = {i: float(recall[i]) for i in range(num_classes)}
        f1_dict = {i: float(f1[i]) for i in range(num_classes)}

        return EvaluationMetrics(
            accuracy=float(accuracy),
            precision_per_class=precision_dict,
            recall_per_class=recall_dict,
            f1_per_class=f1_dict,
            macro_precision=float(macro_precision),
            macro_recall=float(macro_recall),
            macro_f1=float(macro_f1),
            confusion_matrix=cm,
            classification_report=str(report),
        )

    def compute_accuracy(self, y_true: List[int], y_pred: np.ndarray) -> float:
        """Compute accuracy score only.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Accuracy score between 0 and 1.
        """
        return float(accuracy_score(y_true, y_pred))

    def get_confusion_matrix(self, y_true: List[int], y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            NxN confusion matrix as numpy array.
        """
        all_labels = list(range(len(self.class_names)))
        return confusion_matrix(y_true, y_pred, labels=all_labels)
