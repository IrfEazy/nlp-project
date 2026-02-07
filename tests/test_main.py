"""
Tests for the main module.

Tests cover:
- CLI mode selection
- run_baseline_comparison
- run_single_model
- run_custom_workflow_example
"""

import sys
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.evaluator import EvaluationMetrics
from src.main import (
    main,
    run_baseline_comparison,
    run_custom_workflow_example,
    run_single_model,
)


@pytest.fixture
def mock_workflow_components() -> Generator[dict[str, Any], Any, None]:
    """Setup common mocks for workflow testing."""
    with (
        patch("src.main.WorkflowBuilder") as mock_builder,
        patch("src.main.LoadDataStep") as mock_load,
        patch("src.main.TrainModelStep") as mock_train,
        patch("src.main.EvaluateModelStep") as mock_eval,
        patch("src.main.VisualizeResultsStep") as mock_viz,
        patch("src.main.ResultsVisualizer") as mock_visualizer,
    ):

        # Setup mock workflow
        mock_workflow = MagicMock()
        mock_workflow.run.return_value = {
            "train_texts": ["text1", "text2"],
            "train_labels": [0, 1],
            "test_texts": ["test1", "test2"],
            "test_labels": [0, 1],
            "predictions": np.array([0, 1]),
            "trained_strategy": MagicMock(
                get_vocabulary_size=MagicMock(return_value=100),
                get_top_features=MagicMock(return_value={0: ["bad"], 1: ["good"]}),
            ),
            "metrics": EvaluationMetrics(accuracy=0.85),
        }

        # Setup builder chain
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance
        mock_builder_instance.set_name.return_value = mock_builder_instance
        mock_builder_instance.add_step.return_value = mock_builder_instance
        mock_builder_instance.build.return_value = mock_workflow

        yield {
            "builder": mock_builder,
            "workflow": mock_workflow,
            "load_step": mock_load,
            "train_step": mock_train,
            "eval_step": mock_eval,
            "viz_step": mock_viz,
            "visualizer": mock_visualizer,
        }


@pytest.mark.unit
class TestRunBaselineComparison:
    """Tests for run_baseline_comparison function."""

    def test_runs_both_bow_and_tfidf(self, mock_workflow_components) -> None:
        """Should run workflows for both BoW and TF-IDF models."""
        run_baseline_comparison()

        # Should have built at least 2 workflows
        assert mock_workflow_components["builder"].call_count >= 2

    def test_uses_sample_size_when_provided(self, mock_workflow_components) -> None:
        """Should pass sample_size to DataConfig."""
        with patch("src.main.DataConfig") as mock_data_config:
            run_baseline_comparison(sample_size=1000)
            mock_data_config.assert_called_with(sample_size=1000)

    def test_compares_model_results(self, mock_workflow_components) -> None:
        """Should call visualizer.compare_models with results."""
        run_baseline_comparison()

        mock_visualizer_instance = mock_workflow_components["visualizer"].return_value
        mock_visualizer_instance.compare_models.assert_called_once()


@pytest.mark.unit
class TestRunSingleModel:
    """Tests for run_single_model function."""

    def test_runs_workflow(self, mock_workflow_components) -> None:
        """Should build and run a workflow."""
        result = run_single_model()

        mock_workflow_components["workflow"].run.assert_called()
        assert "metrics" in result

    def test_uses_bow_by_default(self, mock_workflow_components) -> None:
        """Should use BoW config by default."""
        with patch("src.main.ExperimentConfig") as mock_config:
            mock_config.for_bow_baseline.return_value = MagicMock()
            mock_config.for_tfidf.return_value = MagicMock()

            run_single_model(model_type="bow")

            mock_config.for_bow_baseline.assert_called()

    def test_uses_tfidf_when_specified(self, mock_workflow_components) -> None:
        """Should use TF-IDF config when specified."""
        with patch("src.main.ExperimentConfig") as mock_config:
            mock_config.for_bow_baseline.return_value = MagicMock()
            mock_config.for_tfidf.return_value = MagicMock()

            run_single_model(model_type="tfidf")

            mock_config.for_tfidf.assert_called()

    def test_returns_context_dict(self, mock_workflow_components) -> None:
        """Should return the workflow result context."""
        result = run_single_model()

        assert isinstance(result, dict)
        assert "metrics" in result
        assert result["metrics"].accuracy == 0.85


@pytest.mark.unit
class TestRunCustomWorkflowExample:
    """Tests for run_custom_workflow_example function."""

    def test_executes_manual_steps(self, capsys) -> None:
        """Should execute pipeline steps manually."""
        with (
            patch("src.main.LoadDataStep") as mock_load,
            patch("src.main.TrainModelStep") as mock_train,
            patch("src.main.EvaluateModelStep") as mock_eval,
            patch("src.main.VisualizeResultsStep") as mock_viz,
        ):

            # Setup mock returns
            mock_load_instance = MagicMock()
            mock_load_instance.execute.return_value = {
                "train_texts": ["text1", "text2"],
                "train_labels": [0, 1],
            }
            mock_load.return_value = mock_load_instance

            mock_train_instance = MagicMock()
            mock_strategy = MagicMock()
            mock_strategy.get_vocabulary_size.return_value = 100
            mock_strategy.get_top_features.return_value = {0: ["word1"], 1: ["word2"]}
            mock_train_instance.execute.return_value = {
                "train_texts": ["text1", "text2"],
                "train_labels": [0, 1],
                "trained_strategy": mock_strategy,
                "predictions": np.array([0, 1]),
            }
            mock_train.return_value = mock_train_instance

            mock_eval_instance = MagicMock()
            mock_eval_instance.execute.return_value = {
                "train_texts": ["text1", "text2"],
                "train_labels": [0, 1],
                "trained_strategy": mock_strategy,
                "predictions": np.array([0, 1]),
                "metrics": EvaluationMetrics(accuracy=0.8),
            }
            mock_eval.return_value = mock_eval_instance

            mock_viz_instance = MagicMock()
            mock_viz_instance.execute.return_value = {}
            mock_viz.return_value = mock_viz_instance

            run_custom_workflow_example()

            # Verify all steps were executed
            mock_load_instance.execute.assert_called_once()
            mock_train_instance.execute.assert_called_once()
            mock_eval_instance.execute.assert_called_once()
            mock_viz_instance.execute.assert_called_once()


@pytest.mark.unit
class TestMain:
    """Tests for main function CLI parsing."""

    def test_baseline_mode_by_default(self, mock_workflow_components) -> None:
        """Should run baseline comparison when no args provided."""
        with (
            patch("src.main.configure_environment"),
            patch.object(sys, "argv", ["main.py"]),
        ):
            main()

            # Baseline runs 2 workflows
            assert mock_workflow_components["workflow"].run.call_count >= 2

    def test_baseline_mode_explicit(self, mock_workflow_components) -> None:
        """Should run baseline comparison for 'baseline' mode."""
        with (
            patch("src.main.configure_environment"),
            patch.object(sys, "argv", ["main.py", "baseline"]),
        ):
            main()

            assert mock_workflow_components["workflow"].run.call_count >= 2

    def test_single_mode(self, mock_workflow_components) -> None:
        """Should run single model for 'single' mode."""
        with (
            patch("src.main.configure_environment"),
            patch.object(sys, "argv", ["main.py", "single"]),
        ):
            main()

            mock_workflow_components["workflow"].run.assert_called()

    def test_custom_mode(self) -> None:
        """Should run custom workflow for 'custom' mode."""
        with (
            patch("src.main.configure_environment"),
            patch("src.main.run_custom_workflow_example") as mock_custom,
            patch.object(sys, "argv", ["main.py", "custom"]),
        ):
            main()

            mock_custom.assert_called_once()

    def test_unknown_mode_exits(self, capsys) -> None:
        """Should exit with error for unknown mode."""
        with (
            patch("src.main.configure_environment"),
            patch.object(sys, "argv", ["main.py", "unknown_mode"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_sample_size_from_args(self, mock_workflow_components) -> None:
        """Should parse sample_size from command line."""
        with (
            patch("src.main.configure_environment"),
            patch("src.main.DataConfig") as mock_data_config,
            patch.object(sys, "argv", ["main.py", "baseline", "5000"]),
        ):
            main()

            # Should have been called with sample_size
            mock_data_config.assert_called_with(sample_size=5000)

    def test_calls_configure_environment(self, mock_workflow_components) -> None:
        """Should configure environment for reproducibility."""
        with (
            patch("src.main.configure_environment") as mock_configure,
            patch.object(sys, "argv", ["main.py", "baseline"]),
        ):
            main()

            mock_configure.assert_called_once_with(seed=42)
            mock_configure.assert_called_once_with(seed=42)
