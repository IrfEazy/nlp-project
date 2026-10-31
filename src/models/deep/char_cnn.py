"""Character-level CNN (Zhang et al. 2015 style) ported from the notebook.

Two alphabet variants: A (70 lowercase characters, text lowercased) and B
(96 characters, case kept). Any other character, including the space, maps to
index 0 (also the padding index). Sequences keep their *last* ``max_length``
characters, are left-padded and then reversed (backward quantization).
"""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

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

ALPHABET_LOWER = (
    "abcdefghijklmnopqrstuvwxyz0123456789—,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
)
ALPHABET_MIXED = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "—,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
)
MAX_LENGTH = 1014
_MAX_CODEPOINT = 0x110000


def build_lookup(alphabet: str) -> NDArray[np.uint8]:
    """Codepoint -> index table (1..len(alphabet)); everything else 0."""
    table = np.zeros(_MAX_CODEPOINT, dtype=np.uint8)
    for i, char in enumerate(alphabet):
        table[ord(char)] = i + 1
    return table


def encode_texts(
    texts: List[str], alphabet: str, lowercase: bool, max_length: int = MAX_LENGTH
) -> NDArray[np.uint8]:
    """Encode texts to a ``(n, max_length)`` uint8 array.

    The last ``max_length`` characters are kept, padded on the left with 0,
    then the row is reversed (latest characters first, padding at the end).
    """
    table = build_lookup(alphabet)
    out = np.zeros((len(texts), max_length), dtype=np.uint8)
    for row, text in enumerate(texts):
        if lowercase:
            text = text.lower()
        codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        codes = table[codepoints[-max_length:]]
        out[row, : len(codes)] = codes[::-1]
    return out


class CharCNN(nn.Module):
    """Six conv layers, three pools and a 1024-1024 classifier head.

    The embedding is a frozen one-hot matrix (row 0 is the zero vector).
    """

    def __init__(
        self,
        alphabet_size: int,
        n_classes: int,
        max_length: int = MAX_LENGTH,
        filters: int = 256,
        hidden: int = 1024,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        one_hot = torch.zeros(alphabet_size + 1, alphabet_size)
        one_hot[1:] = torch.eye(alphabet_size)
        self.embedding = nn.Embedding.from_pretrained(one_hot, freeze=True)

        def conv(c_in: int, k: int) -> List[nn.Module]:
            return [nn.Conv1d(c_in, filters, k), nn.ReLU()]

        self.features = nn.Sequential(
            *conv(alphabet_size, 7), nn.MaxPool1d(3),
            *conv(filters, 7), nn.MaxPool1d(3),
            *conv(filters, 3), *conv(filters, 3), *conv(filters, 3),
            *conv(filters, 3), nn.MaxPool1d(3),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.flatten_size = self.features(
                torch.zeros(1, alphabet_size, max_length)
            ).shape[1]
        if self.flatten_size < 1:
            raise ValueError(f"max_length {max_length} is too short for the conv stack")
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        for module in self.modules():  # Keras defaults: glorot uniform, zero bias
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.classifier(self.features(self.embedding(x).transpose(1, 2)))


class CharCnnStrategy(ModelStrategy):
    """Character CNN, alphabet A (70 characters, lowercased text)."""

    alphabet = ALPHABET_LOWER
    lowercase = True
    name = "Character-level CNN (70-char alphabet)"

    def __init__(
        self,
        model_config: ModelConfig,
        preprocessing_config: PreprocessingConfig,
        training_config: Optional[TrainingConfig] = None,
        max_length: int = MAX_LENGTH,
        filters: int = 256,
        hidden: int = 1024,
    ) -> None:
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.training_config = training_config or TrainingConfig()  # the notebook's values
        self.max_length = max_length
        self.filters = filters
        self.hidden = hidden
        self.model = None
        self.history: List[dict] = []

    def _encode(self, texts: List[str]) -> "torch.Tensor":
        return torch.from_numpy(
            encode_texts(texts, self.alphabet, self.lowercase, self.max_length)
        )

    def train(self, train_texts: List[str], train_labels: List[int]) -> None:
        seed = self.model_config.random_seed
        torch.manual_seed(seed)
        n_classes = int(max(train_labels)) + 1
        self.model = CharCNN(
            len(self.alphabet), n_classes, self.max_length, self.filters, self.hidden
        )
        self.history = fit(
            self.model,
            self._encode(train_texts),
            torch.as_tensor(np.asarray(train_labels), dtype=torch.long),
            self.training_config,
            seed,
        )

    def predict(self, texts: List[str]) -> NDArray[np.int_]:
        if self.model is None:
            raise RuntimeError(
                "Strategy must be trained before prediction. Call train() first."
            )
        return predict_classes(
            self.model, self._encode(texts), self.training_config.batch_size, get_device()
        )

    def get_name(self) -> str:
        return self.name


class CharCnnMixedCaseStrategy(CharCnnStrategy):
    """Character CNN, alphabet B (96 characters, case kept)."""

    alphabet = ALPHABET_MIXED
    lowercase = False
    name = "Character-level CNN (96-char mixed-case alphabet)"
