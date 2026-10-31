"""
High-level runs built from pipeline steps: named experiments and dataset analysis.

The CLI calls these functions; they hold no argument parsing.
"""

from typing import Any, Dict, Optional, Set

import nltk
from nltk.corpus import stopwords

from src.config import DataConfig, ExperimentConfig
from src.data.analyzer import DatasetAnalyzer, DatasetStats
from src.data.loader import DataLoader
from src.pipeline.steps import (
    EvaluateModelStep,
    LoadDataStep,
    TrainModelStep,
    VisualizeResultsStep,
)
from src.pipeline.workflow import WorkflowBuilder

TOP_FEATURES = 10


def run_experiment(
    name: str, sample_size: Optional[int] = None, show_plots: bool = True
) -> Dict[str, Any]:
    """Run a named experiment: load, train, evaluate, summarize, plot.

    Args:
        name: Registered experiment name (``ExperimentConfig.experiment_names``).
        sample_size: Optional first-N limit per split.
        show_plots: Whether to display the confusion matrix.

    Returns:
        The final pipeline context.

    Raises:
        ValueError: If the name is unknown.
        ImportError: If the experiment needs an optional extra that is missing.
    """
    config = ExperimentConfig.named(name)
    config.data.sample_size = sample_size

    workflow = (
        WorkflowBuilder()
        .set_name(name)
        .add_step(LoadDataStep(config))
        .add_step(TrainModelStep(config))
        .add_step(EvaluateModelStep())
        .add_step(VisualizeResultsStep(show_plots=show_plots))
        .build()
    )
    context = workflow.run()

    strategy = context["trained_strategy"]
    if hasattr(strategy, "get_top_features"):
        names = context["class_names"]
        print(f"\nTop {TOP_FEATURES} features per class:")
        for label, features in strategy.get_top_features(n=TOP_FEATURES).items():
            print(f"  {names[label]}: {', '.join(features)}")
    return context


def _english_stopwords() -> Set[str]:
    """Return NLTK English stopwords, downloading the corpus if missing."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def _print_stats(split: str, stats: DatasetStats) -> None:
    print(f"\n{split} set ({stats.num_documents} documents)")
    print(f"  Average characters per document: {stats.avg_document_length:.2f}")
    print(f"  Average unique words per document: {stats.avg_vocabulary_per_doc:.2f}")
    print(f"  Vocabulary size: {stats.total_vocabulary}")
    if stats.vocabulary_without_stopwords is not None:
        print(
            "  Vocabulary size without NLTK English stopwords: "
            f"{stats.vocabulary_without_stopwords}"
        )


def run_analysis(sample_size: Optional[int] = None) -> Dict[str, DatasetStats]:
    """Print dataset statistics for the train and test splits.

    Vocabulary is counted after deleting punctuation, lowercasing and
    whitespace splitting (original notebook cells 15-20).

    Args:
        sample_size: Optional first-N limit per split.

    Returns:
        Mapping of split name ("train", "test") to its statistics.
    """
    loader = DataLoader(DataConfig(sample_size=sample_size))
    analyzer = DatasetAnalyzer(stopwords=_english_stopwords())

    stats = {
        "train": analyzer.compute_stats(loader.load_train()[0]),
        "test": analyzer.compute_stats(loader.load_test()[0]),
    }
    print("\nDATASET ANALYSIS")
    for split, split_stats in stats.items():
        _print_stats(split, split_stats)
    return stats
