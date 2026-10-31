"""
Main entry point for the NLP project.

Usage:
    uv run python -m src.main baseline [sample_size]       # Compare BoW vs TF-IDF (default)
    uv run python -m src.main single [model] [sample_size]  # Single model (bow/tfidf)
    uv run python -m src.main custom                       # Custom workflow example
"""

import sys
from typing import Optional

from .config import DataConfig, ExperimentConfig
from .evaluation.visualizer import ResultsVisualizer
from .models.factory import ModelFactory
from .pipeline import (
    EvaluateModelStep,
    LoadDataStep,
    TrainModelStep,
    VisualizeResultsStep,
    WorkflowBuilder,
)
from .utils import configure_environment


def run_baseline_comparison(sample_size: Optional[int] = None) -> None:
    """Run baseline comparison between BoW and TF-IDF models.

    Trains both models on the same data and compares their performance.

    Args:
        sample_size: Optional sample size for faster iteration.
    """
    # Configure data (optionally with sampling for development)
    data_config = DataConfig(sample_size=sample_size)

    # Run BoW baseline
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON: BoW vs TF-IDF")
    print("=" * 70)

    # Create BoW config
    bow_config = ExperimentConfig.for_bow_baseline()
    bow_config.data = data_config

    # Create TF-IDF config
    tfidf_config = ExperimentConfig.for_tfidf()
    tfidf_config.data = data_config

    # Build workflow for BoW
    bow_workflow = (
        WorkflowBuilder()
        .set_name("BoW + Logistic Regression")
        .add_step(LoadDataStep(bow_config))
        .add_step(TrainModelStep(bow_config))
        .add_step(EvaluateModelStep())
        .build()
    )

    bow_result = bow_workflow.run()
    bow_accuracy = bow_result["metrics"].accuracy

    # Build workflow for TF-IDF (reuse loaded data)
    tfidf_workflow = (
        WorkflowBuilder()
        .set_name("TF-IDF + Logistic Regression")
        .add_step(TrainModelStep(tfidf_config))
        .add_step(EvaluateModelStep())
        .build()
    )

    # Pass data from BoW run
    tfidf_result = tfidf_workflow.run(
        {
            "train_texts": bow_result["train_texts"],
            "train_labels": bow_result["train_labels"],
            "test_texts": bow_result["test_texts"],
            "test_labels": bow_result["test_labels"],
        }
    )
    tfidf_accuracy = tfidf_result["metrics"].accuracy

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    visualizer = ResultsVisualizer()
    visualizer.compare_models(
        model_names=["BoW + LR", "TF-IDF + LR"],
        accuracies=[bow_accuracy, tfidf_accuracy],
        title="Baseline Model Comparison",
    )


def run_single_model(
    model_type: str = "bow", sample_size: Optional[int] = None
) -> dict:
    """Run a single model pipeline.

    Args:
        model_type: Model type ('bow' or 'tfidf').
        sample_size: Optional sample size for faster iteration.

    Returns:
        Context dictionary with results.

    Raises:
        ValueError: If model_type is not an available model.
    """
    available = ModelFactory.available_models()
    if model_type not in available:
        raise ValueError(
            f"Unknown model '{model_type}'. Available models: {', '.join(available)}"
        )

    config = ExperimentConfig.for_model(model_type)

    config.data = DataConfig(sample_size=sample_size)

    # Build and run workflow
    workflow = (
        WorkflowBuilder()
        .set_name(f"{model_type.upper()} Pipeline")
        .add_step(LoadDataStep(config))
        .add_step(TrainModelStep(config))
        .add_step(EvaluateModelStep())
        .add_step(VisualizeResultsStep(show_plots=True))
        .build()
    )

    return workflow.run()


def run_custom_workflow_example() -> None:
    """Demonstrate a custom workflow with manual step composition."""
    print("\n" + "=" * 70)
    print("CUSTOM WORKFLOW EXAMPLE")
    print("=" * 70)

    # Use small sample for demo
    config = ExperimentConfig.for_bow_baseline()
    config.data = DataConfig(sample_size=5000)

    # Manually compose and execute steps
    context = {}

    # Step 1: Load data
    load_step = LoadDataStep(config)
    context = load_step.execute(context)
    print(f"Loaded {len(context['train_texts'])} training samples")

    # Step 2: Train model
    train_step = TrainModelStep(config)
    context = train_step.execute(context)

    strategy = context["trained_strategy"]
    print(f"\nVocabulary size: {strategy.get_vocabulary_size()}")
    print("\nTop features per class:")
    top_features = strategy.get_top_features(n=5)
    for label, features in top_features.items():
        print(f"  Class {label}: {', '.join(features)}")

    # Step 3: Evaluate
    eval_step = EvaluateModelStep()
    context = eval_step.execute(context)

    # Step 4: Visualize (optional)
    viz_step = VisualizeResultsStep(show_plots=True)
    context = viz_step.execute(context)


def main() -> None:
    """Main entry point with command-line mode selection."""
    # Configure environment for reproducibility
    configure_environment(seed=42)

    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"

    args = sys.argv[2:]

    # Run selected mode
    if mode == "baseline":
        sample_size = None
        if args:
            try:
                sample_size = int(args[0])
                print(f"Using sample size: {sample_size}")
            except ValueError:
                pass
        run_baseline_comparison(sample_size=sample_size)
    elif mode == "single":
        model_type = "bow"
        if args and not args[0].isdigit():
            model_type = args.pop(0)
        available = ModelFactory.available_models()
        if model_type not in available:
            print(
                f"Unknown model: {model_type}. Available models: {', '.join(available)}"
            )
            sys.exit(1)
        sample_size = None
        if args:
            try:
                sample_size = int(args[0])
            except ValueError:
                print(f"Invalid sample size: {args[0]} (expected an integer)")
                sys.exit(1)
            print(f"Using sample size: {sample_size}")
        run_single_model(model_type=model_type, sample_size=sample_size)
    elif mode == "custom":
        run_custom_workflow_example()
    else:
        print(f"Unknown mode: {mode}")
        print("\nUsage:")
        print("  python -m src.main baseline [sample_size]        # Compare BoW vs TF-IDF")
        print("  python -m src.main single [model] [sample_size]  # Single model (bow/tfidf)")
        print("  python -m src.main custom                        # Custom workflow example")
        sys.exit(1)


if __name__ == "__main__":
    main()
