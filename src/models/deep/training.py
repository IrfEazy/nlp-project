"""Shared PyTorch training loop for the deep strategies.

Mirrors the Keras workflow of the original notebook: a seeded shuffle of the
training set, the *last* fraction of the shuffled data as validation set
(Keras ``validation_split``), mini-batch Adam training, early stopping on a
monitored metric with best-weight restoration and an optional
ReduceLROnPlateau on the validation loss.

Model inputs are one tensor per sample (``inputs[i]``); the model maps a batch
of inputs to logits. Strategies own their encoding and call ``fit`` and
``predict_classes``.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
        validation_split: Fraction of the shuffled training set held out
            (taken from the end, as Keras does).
        monitor: ``val_accuracy`` (maximised) or ``val_loss`` (minimised).
        patience: Epochs without improvement before stopping.
        restore_best_weights: Reload the best epoch's weights at the end.
        reduce_lr_patience: If set, ReduceLROnPlateau on ``val_loss`` with this
            patience and ``reduce_lr_factor``.
        reduce_lr_factor: Learning-rate multiplier of ReduceLROnPlateau.
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

    def __post_init__(self) -> None:
        if self.monitor not in _MODES:
            raise ValueError(
                f"Unknown monitor '{self.monitor}'. Available: {sorted(_MODES)}"
            )
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


def split_indices(n: int, validation_split: float, seed: int):
    """Seeded shuffle, then the last fraction is validation (Keras semantics)."""
    order = np.random.RandomState(seed).permutation(n)
    split_at = int(n * (1 - validation_split))
    if split_at < 1 or split_at >= n:
        raise ValueError(
            f"Cannot split {n} samples with validation_split={validation_split}: "
            "both parts need at least one sample."
        )
    return order[:split_at], order[split_at:]


def _evaluate(model, inputs, labels, batch_size, device, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            x = inputs[start : start + batch_size].to(device).long()
            y = labels[start : start + batch_size].to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
    return total_loss / len(labels), correct / len(labels)


def fit(
    model: "nn.Module",
    inputs: "torch.Tensor",
    labels: "torch.Tensor",
    config: TrainingConfig,
    seed: int,
    device: Optional["torch.device"] = None,
) -> List[Dict[str, float]]:
    """Train ``model`` in place and return the per-epoch history.

    Args:
        model: Module mapping a batch of inputs to class logits.
        inputs: Integer tensor, one row per sample (kept on the CPU).
        labels: Integer class tensor.
        config: Hyperparameters.
        seed: Seed for the split and the per-epoch shuffles.
        device: Defaults to ``get_device()``.

    Returns:
        One dict per epoch with ``epoch``, ``loss``, ``accuracy``,
        ``val_loss``, ``val_accuracy`` and ``lr``. The model ends with the best
        epoch's weights when ``restore_best_weights`` is set.
    """
    device = device or get_device()
    model.to(device)
    train_idx, val_idx = split_indices(len(labels), config.validation_split, seed)
    train_idx_t = torch.as_tensor(train_idx)
    x_val, y_val = inputs[torch.as_tensor(val_idx)], labels[torch.as_tensor(val_idx)]

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
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
            x = inputs[batch].to(device).long()
            y = labels[batch].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()

        val_loss, val_acc = _evaluate(
            model, x_val, y_val, config.batch_size, device, criterion
        )
        entry = {
            "epoch": epoch + 1,
            "loss": total_loss / len(order),
            "accuracy": correct / len(order),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        print(
            f"Epoch {entry['epoch']}/{config.max_epochs} - "
            f"loss: {entry['loss']:.4f} - accuracy: {entry['accuracy']:.4f} - "
            f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - "
            f"lr: {entry['lr']:.2e}"
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
    inputs: "torch.Tensor",
    batch_size: int,
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """Argmax class per row of ``inputs``."""
    device = device or get_device()
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = inputs[start : start + batch_size].to(device).long()
            out.append(model(x).argmax(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.int64)
