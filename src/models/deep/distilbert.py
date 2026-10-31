"""DistilBERT sequence classification ported from the notebook (optional ``deep`` extra).

Three experiments share one strategy base: full fine-tuning and frozen-encoder
transfer learning on the five star classes (notebook: ignite loop, batch 8,
lr 5e-5, linear decay to 0) and binary polarity fine-tuning (notebook: Hugging
Face ``Trainer``, batch 32, lr 2e-5, weight decay 0.01). Both run through the
shared PyTorch loop in ``training.py``; nothing is pushed to or read from the
Hugging Face hub (the notebook pushed a model with a hard-coded token).

The zero-shot classifier lives in ``zeroshot.py`` (no training).
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from src.config import ModelConfig, PreprocessingConfig
from src.models.deep.training import (
    TrainingConfig,
    fit,
    get_device,
    nn,
    predict_classes,
    torch,
)
from src.models.strategies import ModelStrategy
from src.preprocessing.feature_engineer import delete_punctuation

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "transformers is required for the DistilBERT models. "
        "Install the optional extra with `uv sync --extra deep`."
    ) from exc

CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 512
POLARITY_VALIDATION_SEED = 2307
TOKENIZE_CHUNK = 10_000


class TokenizedTexts:
    """Tokenized texts, padded lazily when a batch is requested.

    ``inputs[index_tensor]`` returns ``{"input_ids", "attention_mask"}`` long
    tensors. ``padding="max_length"`` pads every sample to ``max_length``
    (notebook 5-class runs); ``padding="dynamic"`` pads to the longest sample
    of the batch (``DataCollatorWithPadding``, notebook polarity run). Padding
    lazily avoids holding a ``(n, 512)`` matrix for the whole dataset. Texts
    are tokenized ``chunk_size`` at a time and each row is kept as an int32
    array, so memory stays small for hundreds of thousands of texts.
    """

    def __init__(
        self,
        tokenizer: Any,
        texts: Sequence[str],
        padding: str = "max_length",
        max_length: int = MAX_LENGTH,
        chunk_size: int = TOKENIZE_CHUNK,
    ) -> None:
        if padding not in ("max_length", "dynamic"):
            raise ValueError("padding must be 'max_length' or 'dynamic'")
        self.padding = padding
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self.ids: List[NDArray[np.int32]] = []
        for start in range(0, len(texts), chunk_size):
            chunk = [str(t) for t in texts[start : start + chunk_size]]
            encoded = tokenizer(chunk, truncation=True, max_length=max_length)["input_ids"]
            self.ids.extend(np.asarray(row, dtype=np.int32) for row in encoded)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: "torch.Tensor") -> dict:
        rows = [self.ids[i] for i in torch.as_tensor(index).reshape(-1).tolist()]
        width = self.max_length if self.padding == "max_length" else max(map(len, rows))
        input_ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for row, ids in enumerate(rows):
            input_ids[row, : len(ids)] = torch.from_numpy(ids.astype(np.int64))
            mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class TransformerClassifier(nn.Module):
    """Wraps a Hugging Face sequence classifier: a batch dict in, logits out."""

    def __init__(self, hf_model: "nn.Module") -> None:
        super().__init__()
        self.model = hf_model

    def forward(self, batch: dict) -> "torch.Tensor":
        return self.model(**batch).logits


def freeze_encoder(model: "nn.Module") -> None:
    """Freeze the pretrained ``distilbert`` encoder; the head stays trainable."""
    for param in model.model.distilbert.parameters():
        param.requires_grad = False


def parameter_summary(model: "nn.Module") -> Tuple[int, int]:
    """Return ``(trainable, frozen)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


class TransformerStrategy(ModelStrategy):
    """Fine-tuning strategy for a Hugging Face sequence classification model.

    Subclasses set the class attributes below; ``tokenizer`` and ``model`` may
    be injected (offline tests), otherwise ``checkpoint`` is loaded from the
    Hugging Face cache or downloaded when training starts.
    """

    name = "Transformer"
    checkpoint = CHECKPOINT
    freeze = False
    padding = "max_length"
    clean_punctuation = False

    def __init__(
        self,
        model_config: ModelConfig,
        preprocessing_config: PreprocessingConfig,
        n_classes: Optional[int] = None,
        training_config: Optional[TrainingConfig] = None,
        tokenizer: Any = None,
        model: Any = None,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.max_length = max_length
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.n_classes = n_classes
        self.training_config = training_config or self.default_training_config()
        self.tokenizer = tokenizer
        self._hf_model = model
        self.model: Optional[TransformerClassifier] = None
        self.history: List[dict] = []

    @classmethod
    def default_training_config(cls) -> TrainingConfig:
        raise NotImplementedError

    def _clean(self, texts: List[str]) -> List[str]:
        if self.preprocessing_config.strip_punctuation:
            return delete_punctuation(texts)
        return texts

    def _tokenize(self, texts: List[str]) -> TokenizedTexts:
        return TokenizedTexts(
            self.tokenizer, self._clean(texts), self.padding, self.max_length
        )

    def _build_model(self) -> TransformerClassifier:
        if self.n_classes is None:
            raise ValueError(f"{type(self).__name__} needs n_classes (the task's class count)")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        hf_model = self._hf_model or AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, num_labels=self.n_classes
        )
        model = TransformerClassifier(hf_model)
        if self.freeze:
            freeze_encoder(model)
        trainable, frozen = parameter_summary(model)
        print(f"Trainable parameters: {trainable:,}; frozen parameters: {frozen:,}")
        return model

    def _fit(self, texts: List[str], labels: "torch.Tensor", seed: int) -> List[dict]:
        """Train with the shared loop; the default splits ``validation_split`` off."""
        return fit(self.model, self._tokenize(texts), labels, self.training_config, seed)

    def train(self, train_texts: List[str], train_labels: List[int]) -> None:
        seed = self.model_config.random_seed
        torch.manual_seed(seed)  # the new classification head is initialised randomly
        self.model = self._build_model()
        labels = torch.as_tensor(np.asarray(train_labels), dtype=torch.long)
        self.history = self._fit(train_texts, labels, seed)

    def predict(self, texts: List[str]) -> NDArray[np.int_]:
        if self.model is None:
            raise RuntimeError(
                "Strategy must be trained before prediction. Call train() first."
            )
        return predict_classes(
            self.model, self._tokenize(texts), self.training_config.batch_size, get_device()
        )

    def get_name(self) -> str:
        return self.name


def five_class_training_config() -> TrainingConfig:
    """Notebook ignite loop: AdamW 5e-5, batch 8, 1 epoch, linear decay to 0.

    The training texts are shuffled with the seed and split 0.75/0.25 into
    training and validation; early stopping on validation accuracy (patience
    2); the training part is evaluated after each epoch as in the notebook.
    """
    return TrainingConfig(
        learning_rate=5e-5,
        batch_size=8,
        max_epochs=1,
        validation_split=0.25,
        shuffle_before_split=True,
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        optimizer="adamw",
        weight_decay=0.0,
        linear_decay=True,
        evaluate_train=True,
    )


def polarity_training_config() -> TrainingConfig:
    """Notebook Hugging Face Trainer: AdamW 2e-5, batch 32, weight decay 0.01.

    One epoch, linear decay to 0 without warmup and gradient clipping at 1.0
    (Trainer defaults). The notebook run used two devices (global_step 8125,
    effective batch 64); this runs on one device with batch 32.
    """
    return TrainingConfig(
        learning_rate=2e-5,
        batch_size=32,
        max_epochs=1,
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        optimizer="adamw",
        weight_decay=0.01,
        linear_decay=True,
        max_grad_norm=1.0,
    )


class DistilBertFineTuneStrategy(TransformerStrategy):
    """DistilBERT with all weights trainable (five star classes)."""

    name = "DistilBERT fine-tuning"

    @classmethod
    def default_training_config(cls) -> TrainingConfig:
        return five_class_training_config()


class DistilBertTransferStrategy(DistilBertFineTuneStrategy):
    """DistilBERT with a frozen encoder: only the classification head trains."""

    name = "DistilBERT transfer learning (frozen encoder)"
    freeze = True


class DistilBertPolarityStrategy(TransformerStrategy):
    """DistilBERT fine-tuning for polarity: dynamic padding, 80/20 split.

    The validation split uses ``train_test_split(test_size=0.2,
    random_state=2307)`` of the notebook, on the (punctuation-deleted) texts.
    """

    name = "DistilBERT polarity fine-tuning"
    padding = "dynamic"
    validation_fraction = 0.2
    split_seed = POLARITY_VALIDATION_SEED

    @classmethod
    def default_training_config(cls) -> TrainingConfig:
        return polarity_training_config()

    def _fit(self, texts: List[str], labels: "torch.Tensor", seed: int) -> List[dict]:
        train_x, val_x, train_y, val_y = train_test_split(
            self._clean(texts),
            labels.tolist(),
            test_size=self.validation_fraction,
            random_state=self.split_seed,
        )
        history = fit(
            self.model,
            TokenizedTexts(self.tokenizer, train_x, self.padding, self.max_length),
            torch.as_tensor(train_y, dtype=torch.long),
            self.training_config,
            seed,
            validation=(
                TokenizedTexts(self.tokenizer, val_x, self.padding, self.max_length),
                torch.as_tensor(val_y, dtype=torch.long),
            ),
        )
        print(f"Validation accuracy: {history[-1]['val_accuracy']:.4f}")
        return history
