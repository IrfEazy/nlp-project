"""
Pipeline steps implementing the Composite Pattern.

Each step performs a specific operation and reads/writes from a shared
context dictionary (like middleware in web frameworks).

Context Keys Convention:
- LoadDataStep writes: train_texts, train_labels, test_texts, test_labels
- TrainModelStep reads: train_texts, train_labels; writes: trained_strategy, predictions
- EvaluateModelStep reads: predictions, test_labels; writes: metrics
- VisualizeResultsStep reads: metrics; displays confusion matrix
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.config import ExperimentConfig
from src.data.loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import ResultsVisualizer
from src.models.factory import ModelFactory


class PipelineStep(ABC):
    """Abstract base class for pipeline steps.

    Composite Pattern: Each step is a leaf that can be composed into
    a Workflow. All steps implement the same interface.

    Context Dictionary Pattern:
    - Each step receives a shared context dict
    - Steps READ required keys and WRITE their outputs
    - Enables loose coupling between steps
    """

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline step.

        Args:
            context: Shared context dictionary with data from previous steps.

        Returns:
            Updated context dictionary with this step's outputs added.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get a human-readable name for this step.

        Returns:
            Step name string.
        """
        pass


class LoadDataStep(PipelineStep):
    """Pipeline step that loads training and test data.

    Writes to context:
        - train_texts: List of training document strings
        - train_labels: List of training labels (0-4)
        - test_texts: List of test document strings
        - test_labels: List of test labels (0-4)
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize with experiment configuration.

        Args:
            config: ExperimentConfig containing data loading settings.
        """
        self.config = config

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load train and test data into context.

        Args:
            context: Shared context dictionary.

        Returns:
            Context with train_texts, train_labels, test_texts, test_labels added.
        """
        print(f"Loading data from {self.config.data.dataset_name}...")

        loader = DataLoader(self.config.data)

        train_texts, train_labels = loader.load_train()
        test_texts, test_labels = loader.load_test()

        context["train_texts"] = train_texts
        context["train_labels"] = train_labels
        context["test_texts"] = test_texts
        context["test_labels"] = test_labels

        print(f"Loaded {len(train_texts)} training and {len(test_texts)} test samples")

        return context

    def get_name(self) -> str:
        return "Load Data"


class TrainModelStep(PipelineStep):
    """Pipeline step that trains a model strategy.

    Reads from context:
        - train_texts: Training documents
        - train_labels: Training labels
        - test_texts: Test documents (for prediction)

    Writes to context:
        - trained_strategy: The trained ModelStrategy instance
        - predictions: Predictions on test set
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize with experiment configuration.

        Args:
            config: ExperimentConfig containing model settings.
        """
        self.config = config

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Train model and generate predictions.

        Args:
            context: Shared context with train/test data.

        Returns:
            Context with trained_strategy and predictions added.

        Raises:
            KeyError: If required context keys are missing.
        """
        # Validate required context keys
        required_keys = ["train_texts", "train_labels", "test_texts"]
        for key in required_keys:
            if key not in context:
                raise KeyError(f"TrainModelStep requires '{key}' in context")

        # Create and train strategy
        model_type = self.config.model.model_type
        print(f"Training {model_type} model...")

        strategy = ModelFactory.create(model_type, self.config)
        strategy.train(context["train_texts"], context["train_labels"])

        print(f"Model trained: {strategy.get_name()}")

        # Generate predictions
        print("Generating predictions on test set...")
        predictions = strategy.predict(context["test_texts"])

        context["trained_strategy"] = strategy
        context["predictions"] = predictions

        return context

    def get_name(self) -> str:
        return "Train Model"


class EvaluateModelStep(PipelineStep):
    """Pipeline step that evaluates model predictions.

    Reads from context:
        - predictions: Model predictions
        - test_labels: Ground truth labels

    Writes to context:
        - metrics: EvaluationMetrics object
    """

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        """Initialize with optional class names.

        Args:
            class_names: Human-readable names for classes.
        """
        self.class_names = class_names or ["0", "1", "2", "3", "4"]

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate predictions and compute metrics.

        Args:
            context: Shared context with predictions and labels.

        Returns:
            Context with metrics added.

        Raises:
            KeyError: If required context keys are missing.
        """
        required_keys = ["predictions", "test_labels"]
        for key in required_keys:
            if key not in context:
                raise KeyError(f"EvaluateModelStep requires '{key}' in context")

        print("Evaluating model performance...")

        evaluator = ModelEvaluator(class_names=self.class_names)
        metrics = evaluator.evaluate(context["test_labels"], context["predictions"])

        context["metrics"] = metrics

        print(f"Accuracy: {metrics.accuracy:.4f}")

        return context

    def get_name(self) -> str:
        return "Evaluate Model"


class VisualizeResultsStep(PipelineStep):
    """Pipeline step that visualizes evaluation results.

    Reads from context:
        - metrics: EvaluationMetrics object
        - trained_strategy: (optional) For model name in title

    Displays:
        - Confusion matrix plot
        - Metrics summary
    """

    def __init__(
        self, show_plots: bool = True, class_names: Optional[List[str]] = None
    ) -> None:
        """Initialize visualization settings.

        Args:
            show_plots: Whether to display matplotlib plots.
            class_names: Human-readable names for classes.
        """
        self.show_plots = show_plots
        self.class_names = class_names or ["0", "1", "2", "3", "4"]

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize evaluation results.

        Args:
            context: Shared context with metrics.

        Returns:
            Unchanged context.

        Raises:
            KeyError: If 'metrics' is missing from context.
        """
        if "metrics" not in context:
            raise KeyError("VisualizeResultsStep requires 'metrics' in context")

        metrics = context["metrics"]
        visualizer = ResultsVisualizer(class_names=self.class_names)

        # Get model name if available
        model_name = "Model"
        if "trained_strategy" in context:
            model_name = context["trained_strategy"].get_name()

        # Print summary
        visualizer.print_metrics_summary(metrics, model_name)

        # Plot confusion matrix
        if self.show_plots and metrics.confusion_matrix is not None:
            visualizer.plot_confusion_matrix_heatmap(
                metrics.confusion_matrix,
                title=f"{model_name} - Confusion Matrix",
                show=True,
            )

        return context

    def get_name(self) -> str:
        return "Visualize Results"
