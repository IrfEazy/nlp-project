"""
Label tasks: how raw dataset labels map to the classes a model predicts.

A task is a data-layer concern. ``DataLoader`` applies it to every split
before first-N sampling, so models, evaluator and visualizer only see the
task's labels (0..num_classes-1) and class names.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LabelTask:
    """Mapping from raw dataset labels to task labels.

    Attributes:
        name: Task identifier used in ``DataConfig.task``.
        num_classes: Number of task classes.
        class_names: Names in label order, or None to take them from the
            dataset features (generic digits when the dataset has none).
        mapping: Raw label -> task label. None means identity. A raw label
            absent from the mapping is dropped together with its row.
    """

    name: str
    num_classes: int
    class_names: Optional[Tuple[str, ...]] = None
    mapping: Optional[Dict[int, int]] = None

    def apply(
        self, texts: Sequence[str], labels: Sequence[int]
    ) -> Tuple[List[str], List[int]]:
        """Map labels and drop unmapped rows, preserving order."""
        if self.mapping is None:
            return list(texts), list(labels)
        kept = [(t, self.mapping[y]) for t, y in zip(texts, labels) if y in self.mapping]
        return [t for t, _ in kept], [y for _, y in kept]


_STARS = tuple(str(i) for i in range(5))

TASKS: Dict[str, LabelTask] = {
    task.name: task
    for task in (
        LabelTask("stars5", 5, _STARS),
        LabelTask(
            "stars3",
            3,
            ("negative", "neutral", "positive"),
            {0: 0, 1: 0, 2: 1, 3: 2, 4: 2},
        ),
        LabelTask("polarity", 2, ("negative", "positive"), {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}),
        LabelTask(
            "polarity_no_neutral",
            2,
            ("negative", "positive"),
            {0: 0, 1: 0, 3: 1, 4: 1},
        ),
        LabelTask("dbpedia14", 14),
    )
}


def get_task(name: str) -> LabelTask:
    """Return the task registered under ``name``.

    Raises:
        ValueError: If the task is unknown.
    """
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {sorted(TASKS)}")
    return TASKS[name]
