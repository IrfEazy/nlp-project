"""Zero-shot NLI classification ported from the notebook (optional ``deep`` extra).

The Hugging Face ``zero-shot-classification`` pipeline scores one entailment
hypothesis per candidate label (``hypothesis_template`` filled with the label)
and the top label wins. Nothing is trained: the experiment's candidate labels
(index = class id) and hypothesis template come from ``ModelConfig``, never
from pipeline defaults. ``hypothesis_template=None`` deliberately selects the
installed pipeline's own default template, ``PIPELINE_DEFAULT_TEMPLATE``.
"""

from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray

from src.config import ModelConfig, PreprocessingConfig
from src.models.deep.training import get_device
from src.models.strategies import ModelStrategy

try:
    from transformers import pipeline
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "transformers is required for the zero-shot models. "
        "Install the optional extra with `uv sync --extra deep`."
    ) from exc

CHECKPOINT = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
# Default of ZeroShotClassificationPipeline.preprocess in transformers 5.x.
PIPELINE_DEFAULT_TEMPLATE = "This example is {}."
BATCH_SIZE = 8
CHUNK_SIZE = 512


class ZeroShotStrategy(ModelStrategy):
    """NLI zero-shot classifier; ``train`` is a no-op.

    ``classifier`` may be injected (offline tests); otherwise the pipeline is
    built on first use on CUDA if available, else CPU.
    """

    name = "DeBERTa zero-shot"
    checkpoint = CHECKPOINT

    def __init__(
        self,
        model_config: ModelConfig,
        preprocessing_config: PreprocessingConfig,
        n_classes: Optional[int] = None,
        classifier: Any = None,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        labels = model_config.candidate_labels
        if not labels:
            raise ValueError("Zero-shot needs ModelConfig.candidate_labels (index = class id)")
        if n_classes is not None and len(labels) != n_classes:
            raise ValueError(
                f"{len(labels)} candidate labels for a task with {n_classes} classes"
            )
        if len(set(labels)) != len(labels):
            raise ValueError("Candidate labels must be distinct")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.candidate_labels: List[str] = [str(label) for label in labels]
        self.hypothesis_template = model_config.hypothesis_template
        self.batch_size = batch_size
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.chunk_size = CHUNK_SIZE
        self._classifier = classifier

    def _pipeline(self) -> Any:
        if self._classifier is None:
            device = get_device()
            print(f"Zero-shot model {self.checkpoint} on {device}")
            self._classifier = pipeline(
                "zero-shot-classification", model=self.checkpoint, device=device
            )
        return self._classifier

    def train(self, train_texts: List[str], train_labels: List[int]) -> None:
        """Nothing to train; the pretrained NLI model is used as is."""

    def predict(self, texts: List[str]) -> NDArray[np.int_]:
        if not texts:
            return np.array([], dtype=int)
        options = {"multi_label": False, "batch_size": self.batch_size}
        if self.hypothesis_template is not None:
            options["hypothesis_template"] = self.hypothesis_template
        classifier = self._pipeline()
        index = {label: i for i, label in enumerate(self.candidate_labels)}
        texts = [str(t) for t in texts]
        predictions: List[int] = []
        for start in range(0, len(texts), self.chunk_size):
            results = classifier(
                texts[start : start + self.chunk_size],
                candidate_labels=self.candidate_labels,
                **options,
            )
            if isinstance(results, dict):  # a single text yields one dict
                results = [results]
            predictions.extend(index[r["labels"][0]] for r in results)
            print(f"Zero-shot: {len(predictions)}/{len(texts)} texts")
        return np.array(predictions, dtype=int)

    def get_name(self) -> str:
        return self.name
