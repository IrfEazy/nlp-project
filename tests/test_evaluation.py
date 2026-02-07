"""
Tests for the evaluation module.

Tests cover:
- ModelEvaluator metrics computation
- EvaluationMetrics dataclass
- ResultsVisualizer (without showing plots)
"""

import numpy as np
import pytest

from src.evaluation.evaluator import EvaluationMetrics, ModelEvaluator
from src.evaluation.visualizer import ResultsVisualizer


@pytest.mark.unit
class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_initialization_default(self) -> None:
        """Evaluator should initialize with default class names."""
        evaluator = ModelEvaluator()

        assert evaluator.class_names == ["0", "1", "2", "3", "4"]

    def test_initialization_custom_class_names(self) -> None:
        """Evaluator should accept custom class names."""
        evaluator = ModelEvaluator(class_names=["neg", "neu", "pos"])

        assert evaluator.class_names == ["neg", "neu", "pos"]

    def test_evaluate_returns_metrics(self, sample_labels, sample_predictions) -> None:
        """evaluate should return EvaluationMetrics object."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(sample_labels, sample_predictions)

        assert isinstance(metrics, EvaluationMetrics)

    def test_evaluate_perfect_predictions(self) -> None:
        """Perfect predictions should yield 100% accuracy."""
        evaluator = ModelEvaluator()
        y_true = [0, 1, 2, 3, 4]
        y_pred = np.array([0, 1, 2, 3, 4])

        metrics = evaluator.evaluate(y_true, y_pred)

        assert metrics.accuracy == 1.0

    def test_evaluate_wrong_predictions(self) -> None:
        """Completely wrong predictions should yield 0% accuracy."""
        evaluator = ModelEvaluator()
        y_true = [0, 0, 0, 0, 0]
        y_pred = np.array([1, 1, 1, 1, 1])

        metrics = evaluator.evaluate(y_true, y_pred)

        assert metrics.accuracy == 0.0

    def test_evaluate_confusion_matrix_shape(
        self, sample_labels, sample_predictions
    ) -> None:
        """Confusion matrix should be 5x5 for 5-class problem."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(sample_labels, sample_predictions)

        assert metrics.confusion_matrix is not None
        assert metrics.confusion_matrix.shape == (5, 5)

    def test_evaluate_per_class_metrics(
        self, sample_labels, sample_predictions
    ) -> None:
        """evaluate should compute per-class precision, recall, F1."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(sample_labels, sample_predictions)

        assert len(metrics.precision_per_class) == 5
        assert len(metrics.recall_per_class) == 5
        assert len(metrics.f1_per_class) == 5

        # All values should be between 0 and 1
        for i in range(5):
            assert 0 <= metrics.precision_per_class[i] <= 1
            assert 0 <= metrics.recall_per_class[i] <= 1
            assert 0 <= metrics.f1_per_class[i] <= 1

    def test_evaluate_macro_metrics(self, sample_labels, sample_predictions) -> None:
        """evaluate should compute macro-averaged metrics."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(sample_labels, sample_predictions)

        assert 0 <= metrics.macro_precision <= 1
        assert 0 <= metrics.macro_recall <= 1
        assert 0 <= metrics.macro_f1 <= 1

    def test_evaluate_classification_report(
        self, sample_labels, sample_predictions
    ) -> None:
        """evaluate should generate classification report string."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(sample_labels, sample_predictions)

        assert isinstance(metrics.classification_report, str)
        assert len(metrics.classification_report) > 0

    def test_compute_accuracy(self) -> None:
        """compute_accuracy should return float accuracy."""
        evaluator = ModelEvaluator()
        y_true = [0, 1, 2, 3, 4]
        y_pred = np.array([0, 1, 2, 0, 0])  # 3 correct

        accuracy = evaluator.compute_accuracy(y_true, y_pred)

        assert accuracy == 0.6

    def test_get_confusion_matrix(self) -> None:
        """get_confusion_matrix should return numpy array with all classes."""
        # Use evaluator with 2 class names for simpler binary classification test
        evaluator = ModelEvaluator(class_names=["0", "1"])
        y_true = [0, 0, 1, 1]
        y_pred = np.array([0, 1, 1, 1])

        cm = evaluator.get_confusion_matrix(y_true, y_pred)

        assert isinstance(cm, np.ndarray)
        # For binary case: [[1, 1], [0, 2]]
        assert cm[0, 0] == 1  # True 0, Pred 0
        assert cm[0, 1] == 1  # True 0, Pred 1


@pytest.mark.unit
class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_required_fields(self) -> None:
        """EvaluationMetrics should require accuracy."""
        metrics = EvaluationMetrics(accuracy=0.75)

        assert metrics.accuracy == 0.75

    def test_default_values(self) -> None:
        """EvaluationMetrics should have sensible defaults."""
        metrics = EvaluationMetrics(accuracy=0.5)

        assert metrics.precision_per_class == {}
        assert metrics.recall_per_class == {}
        assert metrics.f1_per_class == {}
        assert metrics.macro_precision == 0.0
        assert metrics.macro_recall == 0.0
        assert metrics.macro_f1 == 0.0
        assert metrics.confusion_matrix is None
        assert metrics.classification_report == ""

    def test_all_fields(self, sample_confusion_matrix) -> None:
        """EvaluationMetrics should store all provided fields."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision_per_class={0: 0.9, 1: 0.8},
            recall_per_class={0: 0.85, 1: 0.75},
            f1_per_class={0: 0.87, 1: 0.77},
            macro_precision=0.85,
            macro_recall=0.80,
            macro_f1=0.82,
            confusion_matrix=sample_confusion_matrix,
            classification_report="test report",
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision_per_class[0] == 0.9
        assert metrics.confusion_matrix is not None
        assert metrics.classification_report == "test report"


@pytest.mark.unit
class TestResultsVisualizer:
    """Tests for ResultsVisualizer class."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self) -> None:
        """Configure matplotlib to use non-interactive backend."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

    def test_initialization_default(self) -> None:
        """Visualizer should initialize with default settings."""
        visualizer = ResultsVisualizer()

        assert visualizer.class_names == ["0", "1", "2", "3", "4"]
        assert visualizer.figsize == (10, 8)

    def test_initialization_custom(self) -> None:
        """Visualizer should accept custom settings."""
        visualizer = ResultsVisualizer(class_names=["a", "b", "c"], figsize=(12, 10))

        assert visualizer.class_names == ["a", "b", "c"]
        assert visualizer.figsize == (12, 10)

    def test_plot_confusion_matrix_returns_figure(
        self, sample_confusion_matrix
    ) -> None:
        """plot_confusion_matrix should return matplotlib Figure."""
        import matplotlib.pyplot as plt

        visualizer = ResultsVisualizer()
        fig = visualizer.plot_confusion_matrix(sample_confusion_matrix, show=False)

        assert fig is not None
        plt.close(fig)

    def test_compare_models_returns_figure(self) -> None:
        """compare_models should return matplotlib Figure."""
        import matplotlib.pyplot as plt

        visualizer = ResultsVisualizer()
        fig = visualizer.compare_models(
            model_names=["Model A", "Model B"], accuracies=[0.75, 0.80], show=False
        )

        assert fig is not None
        plt.close(fig)

    def test_print_metrics_summary(self, capsys, sample_confusion_matrix) -> None:
        """print_metrics_summary should print formatted output."""
        metrics = EvaluationMetrics(
            accuracy=0.75,
            macro_precision=0.70,
            macro_recall=0.65,
            macro_f1=0.67,
            classification_report="Test report content",
        )

        visualizer = ResultsVisualizer()
        visualizer.print_metrics_summary(metrics, "Test Model")

        captured = capsys.readouterr()
        assert "Test Model" in captured.out
        assert "0.75" in captured.out or "75" in captured.out
        assert "Accuracy" in captured.out
