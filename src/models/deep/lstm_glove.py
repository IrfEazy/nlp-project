"""Bidirectional LSTM over frozen GloVe embeddings, ported from the notebook.

Pipeline: ``TextPreprocessor`` cleaning (notebook ``remove_noise``) -> vocabulary
of the words seen at least ``min_word_frequency`` times in the training texts
-> index sequences (out-of-vocabulary words become 0 and keep their position)
truncated/padded to ``max_length`` at the end -> frozen GloVe embedding ->
``BiLSTM(256)`` -> dense 512/256/128 -> classes. Padding is not masked, as in
the notebook's Keras model (no ``mask_zero``).
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
from src.preprocessing.text_preprocessor import TextPreprocessor

EMBEDDING_DIM = 300
MAX_LENGTH = 300
MIN_WORD_FREQUENCY = 50
HIDDEN_SIZE = 256


# Keras Tokenizer default ``filters``: replaced by the split character at encoding
KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
_KERAS_FILTER_TABLE = str.maketrans({char: " " for char in KERAS_FILTERS})


def build_vocabulary(
    cleaned_texts: Sequence[str], min_frequency: int = MIN_WORD_FREQUENCY
) -> Dict[str, int]:
    """Words with at least ``min_frequency`` occurrences, indices from 1.

    Order is the first occurrence in the texts (dict insertion order of the
    word counter, as in the notebook). Index 0 is reserved for padding and
    out-of-vocabulary words.
    """
    counts = Counter(word for text in cleaned_texts for word in text.split())
    common = [word for word, count in counts.items() if count >= min_frequency]
    return {word: index for index, word in enumerate(common, start=1)}


def encode_texts(
    cleaned_texts: Sequence[str], vocabulary: Mapping[str, int], max_length: int = MAX_LENGTH
) -> NDArray[np.int32]:
    """Index sequences of shape ``(n, max_length)``.

    Unknown words map to 0 and keep their position; sequences longer than
    ``max_length`` keep their first words, shorter ones are padded with 0 at
    the end (Keras ``padding='post'``, ``truncating='post'``).

    As in the notebook's ``texts_to_sequences``, the Keras default filter
    characters (including the backslash) are turned into spaces and the text is
    lowercased before splitting, while the vocabulary was built with a plain
    ``split()``: a kept token such as ``\\nworst`` is looked up as ``nworst``.
    """
    out = np.zeros((len(cleaned_texts), max_length), dtype=np.int32)
    for row, text in enumerate(cleaned_texts):
        words = text.lower().translate(_KERAS_FILTER_TABLE).split()[:max_length]
        indices = [vocabulary.get(word, 0) for word in words]
        out[row, : len(indices)] = indices
    return out


def build_embedding_matrix(
    vocabulary: Mapping[str, int], vectors: Any, dim: int = EMBEDDING_DIM
) -> NDArray[np.float32]:
    """``(len(vocabulary) + 1, dim)`` matrix: row 0 and unknown words are zeros.

    ``vectors`` only needs ``word in vectors`` and ``vectors[word]`` (a gensim
    ``KeyedVectors`` or a plain dict).
    """
    matrix = np.zeros((len(vocabulary) + 1, dim), dtype=np.float32)
    for word, index in vocabulary.items():
        if word in vectors:
            matrix[index] = vectors[word]
    return matrix


def load_glove(source: str) -> Any:
    """Load word vectors from a gensim downloader name or a local file.

    A path that exists is read as a gensim ``.kv`` file or, otherwise, as a
    GloVe text file without header. Anything else is a gensim downloader name
    (for example ``glove-wiki-gigaword-300``), downloaded on first use.
    """
    try:
        from gensim.models import KeyedVectors
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "gensim is required for GloVe embeddings. "
            "Install the optional extra with `uv sync --extra deep`."
        ) from exc
    path = Path(source).expanduser()
    if path.is_file():
        if path.suffix == ".kv":
            return KeyedVectors.load(str(path))
        return KeyedVectors.load_word2vec_format(str(path), binary=False, no_header=True)
    import gensim.downloader

    return gensim.downloader.load(source)


class BiLstmClassifier(nn.Module):
    """Frozen embedding, bidirectional LSTM and a 512-256-128 ReLU head.

    The classifier input is the final hidden state of each direction
    (Keras ``return_sequences=False``): the forward state after the last step
    and the backward state after reading the sequence from the end, so 512
    features. Initialisation follows Keras: Glorot-uniform input and dense
    weights, orthogonal recurrent weights, zero biases with the forget-gate
    bias at 1 (``unit_forget_bias``).

    PyTorch's LSTM has two bias vectors per direction (Keras has one), so the
    trainable parameter count is 2048 higher (1,570,309 vs 1,568,261 for five
    classes); the forget-gate bias 1 sits in ``bias_ih`` and ``bias_hh`` is 0.
    """

    def __init__(
        self, embeddings: "torch.Tensor", n_classes: int, hidden_size: int = HIDDEN_SIZE
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.lstm = nn.LSTM(
            embeddings.shape[1], hidden_size, batch_first=True, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        self._init_weights(hidden_size)

    def _init_weights(self, hidden_size: int) -> None:
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight_ih"):
                nn.init.xavier_uniform_(param)
            elif name.startswith("weight_hh"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
                if name.startswith("bias_ih"):  # gate order i, f, g, o
                    param.data[hidden_size : 2 * hidden_size] = 1.0
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        _, (h_n, _) = self.lstm(self.embedding(x))
        return self.head(torch.cat([h_n[0], h_n[1]], dim=1))


def notebook_training_config() -> TrainingConfig:
    """Notebook cells 149-151, with the best checkpoint restored.

    Adam 1e-3, batch 32, 10 epochs, validation = last 20% in dataset order,
    EarlyStopping(val_loss, patience 1), ReduceLROnPlateau(factor 0.1,
    patience 1). The notebook evaluated the last epoch's weights although it
    saved the best checkpoint; this port restores the best val_loss weights.
    """
    return TrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=10,
        validation_split=0.2,
        shuffle_before_split=False,
        monitor="val_loss",
        patience=1,
        restore_best_weights=True,
        reduce_lr_patience=1,
        reduce_lr_factor=0.1,
        reduce_lr_min_delta=1e-4,
    )


class LstmGloveStrategy(ModelStrategy):
    """BiLSTM over frozen GloVe vectors (``ModelConfig.embedding_source``)."""

    name = "BiLSTM + GloVe"

    def __init__(
        self,
        model_config: ModelConfig,
        preprocessing_config: PreprocessingConfig,
        n_classes: Optional[int] = None,
        training_config: Optional[TrainingConfig] = None,
        max_length: int = MAX_LENGTH,
        min_word_frequency: int = MIN_WORD_FREQUENCY,
        hidden_size: int = HIDDEN_SIZE,
        vectors: Any = None,
    ) -> None:
        """
        Args:
            n_classes: Class count of the experiment's task (required to train).
            vectors: Preloaded word vectors; default loads
                ``model_config.embedding_source`` when training starts.
        """
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.n_classes = n_classes
        self.training_config = training_config or notebook_training_config()
        self.max_length = max_length
        self.min_word_frequency = min_word_frequency
        self.hidden_size = hidden_size
        self.vectors = vectors
        self.vocabulary: Dict[str, int] = {}
        self.model = None
        self.history: List[dict] = []
        self._preprocessor: Optional[TextPreprocessor] = None

    def _clean(self, texts: List[str]) -> List[str]:
        if self._preprocessor is None:
            self._preprocessor = TextPreprocessor(self.preprocessing_config)
        return self._preprocessor.preprocess_batch(texts)

    def _encode(self, texts: List[str]) -> "torch.Tensor":
        return torch.from_numpy(
            encode_texts(self._clean(texts), self.vocabulary, self.max_length)
        )

    def train(self, train_texts: List[str], train_labels: List[int]) -> None:
        if self.n_classes is None:
            raise ValueError("LstmGloveStrategy needs n_classes (the task's class count)")
        seed = self.model_config.random_seed
        torch.manual_seed(seed)
        cleaned = self._clean(train_texts)
        self.vocabulary = build_vocabulary(cleaned, self.min_word_frequency)
        print(f"Vocabulary: {len(self.vocabulary)} words (count >= {self.min_word_frequency})")
        vectors = self.vectors
        if vectors is None:
            vectors = load_glove(self.model_config.embedding_source)
        matrix = build_embedding_matrix(self.vocabulary, vectors, EMBEDDING_DIM)
        self.model = BiLstmClassifier(
            torch.from_numpy(matrix), self.n_classes, self.hidden_size
        )
        self.history = fit(
            self.model,
            torch.from_numpy(encode_texts(cleaned, self.vocabulary, self.max_length)),
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
