"""Shared PyTorch training loop for the deep strategies.

Mirrors the Keras workflow of the original notebook: optionally a seeded
shuffle of the training set, the *last* fraction as validation set (Keras
``validation_split``), mini-batch Adam training, early stopping on a monitored
metric with best-weight restoration and an optional ReduceLROnPlateau on the
validation loss with Keras semantics (absolute ``min_delta``, reduction after
``patience`` epochs without improvement).

Model inputs are indexable by an integer tensor (``inputs[index_tensor]``) and
yield a batch: either an integer tensor or a dict of tensors (tokenized text).
The model maps a batch to logits. Strategies own their encoding and call
``fit`` and ``predict_classes``.

Transformer fine-tuning needs a few more options, all off by default so the
Keras-style behaviour of the char-CNN and the LSTM is unchanged: AdamW with
weight decay, a linear learning-rate decay to 0 over all training steps
(updated every iteration, no warmup), gradient-norm clipping and a per-epoch
evaluation on the training part.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "PyTorch is required for the deep-learning models. "
        "Install the optional extra with `uv sync --extra deep`."
    ) from exc


@dataclass
class TrainingConfig:
    """Hyperparameters of the shared training loop.

    Attributes:
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        max_epochs: Maximum number of epochs.
        validation_split: Fraction of the training set held out (taken from
            the end, as Keras does).
        shuffle_before_split: Seeded shuffle before taking the validation
            part; False keeps the dataset order (Keras ``validation_split``
            without a pre-shuffle).
        monitor: ``val_accuracy`` (maximised) or ``val_loss`` (minimised).
        patience: Epochs without improvement before stopping.
        restore_best_weights: Reload the best epoch's weights at the end.
        reduce_lr_patience: If set (>= 1), ReduceLROnPlateau on ``val_loss``:
            the rate is reduced after this many epochs without improvement
            (Keras ``patience``).
        reduce_lr_factor: Learning-rate multiplier of ReduceLROnPlateau.
        reduce_lr_min_delta: Absolute minimum improvement (Keras ``min_delta``).
        optimizer: ``adam`` or ``adamw`` (decoupled weight decay).
        weight_decay: Weight decay; parameters with fewer than two dimensions
            (biases, layer norms) are excluded, as Hugging Face's Trainer does.
        linear_decay: Decay the rate linearly from ``learning_rate`` to 0 over
            ``max_epochs`` times the steps per epoch, stepped every iteration. The
            logged ``lr`` is read after the last step, so it is 0 at the end.
        max_grad_norm: Clip the gradient norm to this value when set.
        evaluate_train: Also evaluate on the training part after each epoch
            (``train_eval_accuracy`` in the history).
    """

    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 10
    validation_split: float = 0.2
    monitor: str = "val_accuracy"
    patience: int = 4
    restore_best_weights: bool = True
    reduce_lr_patience: Optional[int] = None
    reduce_lr_factor: float = 0.1
    reduce_lr_min_delta: float = 1e-4
    shuffle_before_split: bool = True
    optimizer: str = "adam"
    weight_decay: float = 0.0
    linear_decay: bool = False
    max_grad_norm: Optional[float] = None
    evaluate_train: bool = False

    def __post_init__(self) -> None:
        if self.reduce_lr_patience is not None and self.reduce_lr_patience < 1:
            raise ValueError("reduce_lr_patience must be at least 1")
        if self.monitor not in _MODES:
            raise ValueError(
                f"Unknown monitor '{self.monitor}'. Available: {sorted(_MODES)}"
            )
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1 (exclusive)")


_MODES = {"val_accuracy": "max", "val_loss": "min"}


class EarlyStopping:
    """Keras-style early stopping with best-weight snapshot (strict improvement)."""

    def __init__(self, monitor: str, patience: int) -> None:
        self.mode = _MODES[monitor]
        self.patience = patience
        self.best: Optional[float] = None
        self.best_epoch = -1
        self.best_state: Optional[Dict[str, Any]] = None
        self.wait = 0

    def update(self, epoch: int, value: float, model: "nn.Module") -> bool:
        """Record an epoch result; return True when training should stop."""
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )
        if improved:
            self.best, self.best_epoch, self.wait = value, epoch, 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore(self, model: "nn.Module") -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def get_device() -> "torch.device":
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_indices(n: int, validation_split: float, seed: int, shuffle: bool = True):
    """Optional seeded shuffle, then the last fraction is validation (Keras)."""
    order = np.random.RandomState(seed).permutation(n) if shuffle else np.arange(n)
    split_at = int(n * (1 - validation_split))
    if split_at < 1 or split_at >= n:
        raise ValueError(
            f"Cannot split {n} samples with validation_split={validation_split}: "
            "both parts need at least one sample."
        )
    return order[:split_at], order[split_at:]


Batch = Union["torch.Tensor", Dict[str, "torch.Tensor"]]


def _to_device(batch: Batch, device: "torch.device") -> Batch:
    """Move a tensor batch (as long ids) or a dict of tensors to the device."""
    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    return batch.to(device).long()


def _evaluate(model, inputs, labels, indices, batch_size, device, criterion):
    """Mean loss and accuracy over ``indices`` of ``inputs``/``labels``."""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            x = _to_device(inputs[batch], device)
            y = labels[batch].to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
    return total_loss / len(indices), correct / len(indices)


def _build_optimizer(model: "nn.Module", config: TrainingConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate)
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        [g for g in groups if g["params"]], lr=config.learning_rate
    )


def fit(
    model: "nn.Module",
    inputs: "torch.Tensor",
    labels: "torch.Tensor",
    config: TrainingConfig,
    seed: int,
    device: Optional["torch.device"] = None,
    validation: Optional[Tuple[Any, "torch.Tensor"]] = None,
) -> List[Dict[str, float]]:
    """Train ``model`` in place and return the per-epoch history.

    Args:
        model: Module mapping a batch of inputs to class logits.
        inputs: Integer tensor (one row per sample, kept on the CPU) or any
            object indexable by an index tensor that yields a tensor or dict
            of tensors.
        labels: Integer class tensor.
        config: Hyperparameters.
        seed: Seed for the split and the per-epoch shuffles.
        device: Defaults to ``get_device()``.
        validation: Optional ``(inputs, labels)`` used as the validation set;
            then all of ``inputs`` is training data and ``validation_split``
            and ``shuffle_before_split`` are ignored.

    Returns:
        One dict per epoch with ``epoch``, ``loss``, ``accuracy``,
        ``val_loss``, ``val_accuracy`` and ``lr``. The model ends with the best
        epoch's weights when ``restore_best_weights`` is set.
    """
    device = device or get_device()
    model.to(device)
    print(f"Training on device: {device}")
    if validation is None:
        train_idx, val_idx = split_indices(
            len(labels), config.validation_split, seed, config.shuffle_before_split
        )
        train_idx_t, val_idx_t = torch.as_tensor(train_idx), torch.as_tensor(val_idx)
        x_val, y_val = inputs, labels
    else:
        x_val, y_val = validation
        train_idx_t, val_idx_t = torch.arange(len(labels)), torch.arange(len(y_val))

    optimizer = _build_optimizer(model, config)
    steps_per_epoch = -(-len(train_idx_t) // config.batch_size)
    total_steps = steps_per_epoch * config.max_epochs
    linear = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, 1.0 - step / total_steps)
        )
        if config.linear_decay
        else None
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.reduce_lr_factor,
            # torch reduces after more than `patience` bad epochs, Keras after `patience`.
            patience=config.reduce_lr_patience - 1,
            threshold=config.reduce_lr_min_delta,
            threshold_mode="abs",
        )
        if config.reduce_lr_patience is not None
        else None
    )
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(config.monitor, config.patience)
    generator = torch.Generator().manual_seed(seed)
    history: List[Dict[str, float]] = []

    for epoch in range(config.max_epochs):
        model.train()
        order = train_idx_t[torch.randperm(len(train_idx_t), generator=generator)]
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            x = _to_device(inputs[batch], device)
            y = labels[batch].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if linear is not None:
                linear.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()

        val_loss, val_acc = _evaluate(
            model, x_val, y_val, val_idx_t, config.batch_size, device, criterion
        )
        entry = {
            "epoch": epoch + 1,
            "loss": total_loss / len(order),
            "accuracy": correct / len(order),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if config.evaluate_train:
            _, entry["train_eval_accuracy"] = _evaluate(
                model, inputs, labels, train_idx_t, config.batch_size, device, criterion
            )
        history.append(entry)
        print(
            f"Epoch {entry['epoch']}/{config.max_epochs} - "
            f"loss: {entry['loss']:.4f} - accuracy: {entry['accuracy']:.4f} - "
            f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - "
            f"lr: {entry['lr']:.2e}"
            + (
                f" - train_eval_accuracy: {entry['train_eval_accuracy']:.4f}"
                if config.evaluate_train
                else ""
            )
        )
        if scheduler is not None:
            scheduler.step(val_loss)
        if stopper.update(epoch, entry[config.monitor], model):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if config.restore_best_weights:
        stopper.restore(model)
        print(f"Restored best weights from epoch {stopper.best_epoch + 1}")
    return history


def predict_classes(
    model: "nn.Module",
    inputs: Any,
    batch_size: int,
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """Argmax class per sample of ``inputs`` (tensor or batch-indexable object)."""
    device = device or get_device()
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.arange(start, min(start + batch_size, len(inputs)))
            x = _to_device(inputs[batch], device)
            out.append(model(x).argmax(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.int64)
