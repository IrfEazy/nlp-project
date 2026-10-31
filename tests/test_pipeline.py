"""
Tests for the pipeline module.

Tests cover:
- Pipeline steps (LoadDataStep, TrainModelStep, etc.)
- Workflow composition and execution
- WorkflowBuilder fluent API
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import ExperimentConfig
from src.evaluation.evaluator import EvaluationMetrics
from src.pipeline.runner import run_analysis, run_experiment
from src.pipeline.steps import (
    EvaluateModelStep,
    LoadDataStep,
    PipelineStep,
    TrainModelStep,
    VisualizeResultsStep,
)
from src.pipeline.workflow import Workflow, WorkflowBuilder


@pytest.mark.unit
class TestPipelineSteps:
    """Tests for individual pipeline steps."""

    class TestLoadDataStep:
        """Tests for LoadDataStep."""

        @patch("src.pipeline.steps.DataLoader")
        def test_execute_loads_data(
            self, mock_loader_class, default_experiment_config
        ) -> None:
            """LoadDataStep should load train and test data."""
            mock_loader = MagicMock()
            mock_loader.load_train.return_value = (["text1", "text2"], [0, 1])
            mock_loader.load_test.return_value = (["test1"], [2])
            mock_loader_class.return_value = mock_loader

            step = LoadDataStep(default_experiment_config)
            context = step.execute({})

            assert "train_texts" in context
            assert "train_labels" in context
            assert "test_texts" in context
            assert "test_labels" in context

        @patch("src.pipeline.steps.DataLoader")
        def test_execute_returns_context(
            self, mock_loader_class, default_experiment_config
        ) -> None:
            """LoadDataStep should return updated context."""
            mock_loader = MagicMock()
            mock_loader.load_train.return_value = (["text1"], [0])
            mock_loader.load_test.return_value = (["test1"], [1])
            mock_loader_class.return_value = mock_loader

            step = LoadDataStep(default_experiment_config)
            context = step.execute({"existing_key": "value"})

            # Should preserve existing context
            assert context["existing_key"] == "value"
            # Should add new keys
            assert "train_texts" in context

        def test_get_name(self, default_experiment_config) -> None:
            """LoadDataStep should return meaningful name."""
            step = LoadDataStep(default_experiment_config)

            assert step.get_name() == "Load Data"

    class TestTrainModelStep:
        """Tests for TrainModelStep."""

        @patch("src.pipeline.steps.ModelFactory")
        def test_execute_trains_model(
            self, mock_factory, default_experiment_config
        ) -> None:
            """TrainModelStep should train model strategy."""
            mock_strategy = MagicMock()
            mock_strategy.get_name.return_value = "Test Strategy"
            mock_strategy.predict.return_value = np.array([0, 1])
            mock_factory.create.return_value = mock_strategy

            step = TrainModelStep(default_experiment_config)
            context = {
                "train_texts": ["text1", "text2"],
                "train_labels": [0, 1],
                "test_texts": ["test1", "test2"],
            }

            result = step.execute(context)

            mock_strategy.train.assert_called_once_with(["text1", "text2"], [0, 1])
            assert "trained_strategy" in result
            assert "predictions" in result

        def test_execute_missing_context_raises_error(
            self, default_experiment_config
        ) -> None:
            """TrainModelStep should raise error if required keys missing."""
            step = TrainModelStep(default_experiment_config)

            with pytest.raises(KeyError, match="train_texts"):
                step.execute({})

        def test_get_name(self, default_experiment_config) -> None:
            """TrainModelStep should return meaningful name."""
            step = TrainModelStep(default_experiment_config)

            assert step.get_name() == "Train Model"

    class TestTrainModelStepSingleClass:
        def test_single_class_raises_before_fitting(self, default_experiment_config) -> None:
            context = {"train_texts": ["a", "b"], "train_labels": [3, 3], "test_texts": ["c"]}
            with patch("src.pipeline.steps.ModelFactory") as factory:
                with pytest.raises(ValueError, match=r"1 class.*sample_size"):
                    TrainModelStep(default_experiment_config).execute(context)
            factory.create.assert_not_called()

    class TestEvaluateModelStep:
        """Tests for EvaluateModelStep."""

        def test_execute_computes_metrics(self) -> None:
            """EvaluateModelStep should compute evaluation metrics."""
            step = EvaluateModelStep()
            context = {
                "predictions": np.array([0, 1, 2, 3, 4]),
                "test_labels": [0, 1, 2, 3, 4],
            }

            result = step.execute(context)

            assert "metrics" in result
            assert isinstance(result["metrics"], EvaluationMetrics)
            assert result["metrics"].accuracy == 1.0  # Perfect predictions

        def test_execute_missing_predictions_raises_error(self) -> None:
            """EvaluateModelStep should raise error if predictions missing."""
            step = EvaluateModelStep()

            with pytest.raises(KeyError, match="predictions"):
                step.execute({"test_labels": [0, 1, 2]})

        def test_execute_missing_labels_raises_error(self) -> None:
            """EvaluateModelStep should raise error if labels missing."""
            step = EvaluateModelStep()

            with pytest.raises(KeyError, match="test_labels"):
                step.execute({"predictions": np.array([0, 1, 2])})

        def test_get_name(self) -> None:
            """EvaluateModelStep should return meaningful name."""
            step = EvaluateModelStep()

            assert step.get_name() == "Evaluate Model"

    class TestVisualizeResultsStep:
        """Tests for VisualizeResultsStep."""

        def test_execute_with_metrics(self, sample_confusion_matrix) -> None:
            """VisualizeResultsStep should process metrics without error."""
            metrics = EvaluationMetrics(
                accuracy=0.75,
                confusion_matrix=sample_confusion_matrix,
                classification_report="test report",
            )

            step = VisualizeResultsStep(show_plots=False)  # Don't show plots in tests
            context = {"metrics": metrics}

            result = step.execute(context)

            # Should return context unchanged
            assert result["metrics"] is metrics

        def test_execute_missing_metrics_raises_error(self) -> None:
            """VisualizeResultsStep should raise error if metrics missing."""
            step = VisualizeResultsStep(show_plots=False)

            with pytest.raises(KeyError, match="metrics"):
                step.execute({})

        def test_get_name(self) -> None:
            """VisualizeResultsStep should return meaningful name."""
            step = VisualizeResultsStep()

            assert step.get_name() == "Visualize Results"


@pytest.mark.unit
class TestWorkflow:
    """Tests for Workflow class."""

    def test_initialization(self) -> None:
        """Workflow should initialize with steps and name."""
        step1 = MagicMock(spec=PipelineStep)
        step2 = MagicMock(spec=PipelineStep)

        workflow = Workflow(steps=[step1, step2], name="Test Workflow")

        assert len(workflow.steps) == 2
        assert workflow.name == "Test Workflow"

    def test_run_executes_all_steps(self) -> None:
        """Workflow.run should execute all steps in order."""
        step1 = MagicMock(spec=PipelineStep)
        step1.execute.return_value = {"step1_key": "value1"}
        step1.get_name.return_value = "Step 1"

        step2 = MagicMock(spec=PipelineStep)
        step2.execute.return_value = {"step1_key": "value1", "step2_key": "value2"}
        step2.get_name.return_value = "Step 2"

        workflow = Workflow(steps=[step1, step2])
        _ = workflow.run()

        step1.execute.assert_called_once()
        step2.execute.assert_called_once()

    def test_run_passes_context_between_steps(self) -> None:
        """Workflow should pass context from step to step."""
        step1 = MagicMock(spec=PipelineStep)
        step1.execute.return_value = {"key1": "value1"}
        step1.get_name.return_value = "Step 1"

        step2 = MagicMock(spec=PipelineStep)
        step2.execute.return_value = {"key1": "value1", "key2": "value2"}
        step2.get_name.return_value = "Step 2"

        workflow = Workflow(steps=[step1, step2])
        workflow.run()

        # Step 2 should receive output from Step 1
        step2.execute.assert_called_once_with({"key1": "value1"})

    def test_run_with_initial_context(self) -> None:
        """Workflow.run should accept initial context."""
        step = MagicMock(spec=PipelineStep)
        step.execute.return_value = {"initial": True, "new": "value"}
        step.get_name.return_value = "Step"

        workflow = Workflow(steps=[step])
        _ = workflow.run(initial_context={"initial": True})

        step.execute.assert_called_with({"initial": True})

    def test_run_returns_final_context(self) -> None:
        """Workflow.run should return final context."""
        step = MagicMock(spec=PipelineStep)
        step.execute.return_value = {"result": "final"}
        step.get_name.return_value = "Step"

        workflow = Workflow(steps=[step])
        result = workflow.run()

        assert result == {"result": "final"}

    def test_add_step(self) -> None:
        """Workflow.add_step should append step."""
        workflow = Workflow(steps=[])
        step = MagicMock(spec=PipelineStep)

        workflow.add_step(step)

        assert len(workflow) == 1
        assert workflow.steps[0] is step

    def test_len(self) -> None:
        """len(workflow) should return number of steps."""
        steps: list[PipelineStep] = [MagicMock(spec=PipelineStep) for _ in range(3)]
        workflow = Workflow(steps=steps)

        assert len(workflow) == 3


@pytest.mark.unit
class TestWorkflowBuilder:
    """Tests for WorkflowBuilder class."""

    def test_initialization(self) -> None:
        """WorkflowBuilder should initialize empty."""
        builder = WorkflowBuilder()

        assert len(builder._steps) == 0
        assert builder._name == "Workflow"

    def test_set_name_returns_self(self) -> None:
        """set_name should return self for chaining."""
        builder = WorkflowBuilder()
        result = builder.set_name("My Workflow")

        assert result is builder
        assert builder._name == "My Workflow"

    def test_add_step_returns_self(self) -> None:
        """add_step should return self for chaining."""
        builder = WorkflowBuilder()
        step = MagicMock(spec=PipelineStep)
        result = builder.add_step(step)

        assert result is builder
        assert len(builder._steps) == 1

    def test_build_returns_workflow(self) -> None:
        """build should return configured Workflow."""
        step = MagicMock(spec=PipelineStep)

        workflow = WorkflowBuilder().set_name("Test Workflow").add_step(step).build()

        assert isinstance(workflow, Workflow)
        assert workflow.name == "Test Workflow"
        assert len(workflow) == 1

    def test_build_empty_raises_error(self) -> None:
        """build with no steps should raise ValueError."""
        builder = WorkflowBuilder()

        with pytest.raises(ValueError, match="no steps"):
            builder.build()

    def test_fluent_api_chain(self, default_experiment_config) -> None:
        """Builder should support fluent API chaining."""
        step1 = MagicMock(spec=PipelineStep)
        step2 = MagicMock(spec=PipelineStep)
        step3 = MagicMock(spec=PipelineStep)

        workflow = (
            WorkflowBuilder()
            .set_name("Fluent Workflow")
            .add_step(step1)
            .add_step(step2)
            .add_step(step3)
            .build()
        )

        assert workflow.name == "Fluent Workflow"
        assert len(workflow) == 3

    def test_reset_clears_state(self) -> None:
        """reset should clear builder state."""
        builder = (
            WorkflowBuilder().set_name("Name").add_step(MagicMock(spec=PipelineStep))
        )

        builder.reset()

        assert len(builder._steps) == 0
        assert builder._name == "Workflow"

    def test_reset_returns_self(self) -> None:
        """reset should return self for chaining."""
        builder = WorkflowBuilder()
        result = builder.reset()

        assert result is builder


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for complete pipelines."""

    @patch("src.pipeline.steps.DataLoader")
    @patch("src.pipeline.steps.ModelFactory")
    def test_full_pipeline_execution(
        self, mock_factory, mock_loader_class, default_experiment_config
    ) -> None:
        """Full pipeline should execute all steps and produce metrics."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader.load_train.return_value = (["text1", "text2"], [0, 1])
        mock_loader.load_test.return_value = (["test1", "test2"], [0, 1])
        mock_loader.class_names = ["0", "1", "2", "3", "4"]
        mock_loader_class.return_value = mock_loader

        mock_strategy = MagicMock()
        mock_strategy.get_name.return_value = "Mock Strategy"
        mock_strategy.predict.return_value = np.array([0, 1])
        mock_factory.create.return_value = mock_strategy

        # Build and run pipeline
        workflow = (
            WorkflowBuilder()
            .set_name("Integration Test Pipeline")
            .add_step(LoadDataStep(default_experiment_config))
            .add_step(TrainModelStep(default_experiment_config))
            .add_step(EvaluateModelStep())
            .build()
        )

        result = workflow.run()

        # Verify results
        assert "train_texts" in result
        assert "trained_strategy" in result
        assert "predictions" in result
        assert "metrics" in result
        assert result["metrics"].accuracy == 1.0  # Perfect predictions in mock
        assert result["metrics"].accuracy == 1.0  # Perfect predictions in mock


@pytest.mark.unit
class TestClassNamesFromContext:
    """Evaluate/Visualize steps take class names from the loaded task."""

    def test_evaluate_uses_context_class_names(self) -> None:
        context = {
            "predictions": np.array([0, 1, 1]),
            "test_labels": [0, 1, 0],
            "class_names": ["neg", "pos"],
        }
        result = EvaluateModelStep().execute(context)
        assert len(result["metrics"].confusion_matrix) == 2
        assert "neg" in result["metrics"].classification_report

    def test_evaluate_explicit_names_win(self) -> None:
        context = {
            "predictions": np.array([0, 1]),
            "test_labels": [0, 1],
            "class_names": ["neg", "pos"],
        }
        result = EvaluateModelStep(class_names=["a", "b"]).execute(context)
        assert "a" in result["metrics"].classification_report

    def test_evaluate_defaults_to_five_stars(self) -> None:
        context = {"predictions": np.array([0, 4]), "test_labels": [0, 4]}
        result = EvaluateModelStep().execute(context)
        assert result["metrics"].confusion_matrix.shape == (5, 5)

    def test_load_step_writes_class_names(self) -> None:
        with patch("src.pipeline.steps.DataLoader") as mock_loader:
            loader = mock_loader.return_value
            loader.load_train.return_value = (["a"], [0])
            loader.load_test.return_value = (["b"], [1])
            loader.class_names = ["neg", "pos"]
            context = LoadDataStep(ExperimentConfig()).execute({})
        assert context["class_names"] == ["neg", "pos"]


@pytest.mark.unit
class TestRunner:
    """run_experiment and run_analysis with mocked data."""

    def test_run_experiment_end_to_end(self, capsys) -> None:
        texts = ["good great fine", "bad awful poor", "so so okay"] * 100
        labels = [4, 0, 2] * 100
        with patch("src.data.loader.load_dataset") as mock_load:
            mock_load.return_value = {"text": texts, "label": labels}
            context = run_experiment("lr-bow-no-neutral", sample_size=120, show_plots=False)
        assert context["class_names"] == ["negative", "positive"]
        assert set(context["train_labels"]) == {0, 1}
        assert len(context["train_texts"]) == 120  # label 2 dropped before first-N
        out = capsys.readouterr().out
        assert "Top 10 features per class" in out
        assert "Classification Report" in out

    def test_run_experiment_unknown_name(self) -> None:
        with pytest.raises(ValueError):
            run_experiment("nope")

    def test_run_analysis(self, capsys) -> None:
        with (
            patch("src.data.loader.load_dataset") as mock_load,
            patch("src.pipeline.runner._english_stopwords", return_value={"the"}),
        ):
            mock_load.return_value = {"text": ["The cat, sat", "the dog"], "label": [0, 1]}
            stats = run_analysis()
        assert stats["train"].total_vocabulary == 4
        assert stats["train"].vocabulary_without_stopwords == 3
        assert "Vocabulary size without NLTK" in capsys.readouterr().out
