"""
Results visualization utilities.

Provides visualization methods for model evaluation results including
confusion matrices, accuracy comparisons, and other plots.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
from sklearn.metrics import ConfusionMatrixDisplay

from src.evaluation.evaluator import EvaluationMetrics


class ResultsVisualizer:
    """Visualizes model evaluation results.

    Single Responsibility: Only visualize results, does NOT compute
    metrics (that's ModelEvaluator's job).

    Example:
        >>> visualizer = ResultsVisualizer()
        >>> visualizer.plot_confusion_matrix(metrics.confusion_matrix)
        >>> visualizer.compare_models(model_metrics_list)
    """

    def __init__(
        self, class_names: Optional[List[str]] = None, figsize: tuple = (10, 8)
    ) -> None:
        """Initialize the visualizer.

        Args:
            class_names: Labels for confusion matrix axes.
            figsize: Default figure size for plots.
        """
        self.class_names = class_names or ["0", "1", "2", "3", "4"]
        self.figsize = figsize

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        show: bool = True,
    ) -> figure.Figure:
        """Plot a confusion matrix.

        Args:
            cm: NxN confusion matrix array.
            title: Plot title.
            cmap: Colormap name.
            show: Whether to call plt.show().

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.class_names
        )
        disp.plot(ax=ax, cmap=cmap)
        ax.set_title(title)

        if show:
            plt.show()

        return fig

    def plot_confusion_matrix_heatmap(
        self, cm: np.ndarray, title: str = "Confusion Matrix", show: bool = True
    ) -> figure.Figure:
        """Plot confusion matrix as a seaborn-style heatmap.

        Args:
            cm: NxN confusion matrix array.
            title: Plot title.
            show: Whether to call plt.show().

        Returns:
            Matplotlib Figure object.
        """
        try:
            import seaborn as sns

            fig, ax = plt.subplots(figsize=self.figsize)
            sns.heatmap(
                cm.T,
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cmap="Blues",
                annot=True,
                fmt="d",
                ax=ax,
            )
            ax.set_xlabel("True labels")
            ax.set_ylabel("Predicted labels")
            ax.set_title(title)

            if show:
                plt.show()

            return fig
        except ImportError:
            # Fallback to standard confusion matrix plot
            return self.plot_confusion_matrix(cm, title, show=show)

    def compare_models(
        self,
        model_names: List[str],
        accuracies: List[float],
        title: str = "Model Comparison",
        show: bool = True,
    ) -> figure.Figure:
        """Create a bar chart comparing model accuracies.

        Args:
            model_names: Names of models being compared.
            accuracies: Accuracy scores for each model.
            title: Plot title.
            show: Whether to call plt.show().

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.get_cmap("Blues")(np.linspace(0.4, 0.8, len(model_names)))
        bars = ax.bar(model_names, accuracies, color=colors)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.2, color="r", linestyle="--", alpha=0.5, label="Random (20%)")
        ax.legend()

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def print_metrics_summary(
        self, metrics: EvaluationMetrics, model_name: str = "Model"
    ) -> None:
        """Print a formatted summary of evaluation metrics.

        Args:
            metrics: EvaluationMetrics object.
            model_name: Name to display in the header.
        """
        print(f"\n{'='*60}")
        print(f"{model_name} Evaluation Results")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print(f"Macro Precision: {metrics.macro_precision:.4f}")
        print(f"Macro Recall: {metrics.macro_recall:.4f}")
        print(f"Macro F1: {metrics.macro_f1:.4f}")
        print("\nClassification Report:")
        print(metrics.classification_report)
