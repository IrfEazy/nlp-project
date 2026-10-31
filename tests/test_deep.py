"""Tests for the optional PyTorch models (skipped when torch is absent)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import ExperimentConfig, ModelConfig, PreprocessingConfig  # noqa: E402
from src.models.deep import char_cnn  # noqa: E402
from src.models.deep.char_cnn import (  # noqa: E402
    ALPHABET_LOWER,
    ALPHABET_MIXED,
    CharCNN,
    CharCnnMixedCaseStrategy,
    CharCnnStrategy,
    encode_texts,
)
from src.models.deep.training import (  # noqa: E402
    EarlyStopping,
    TrainingConfig,
    fit,
    split_indices,
)
from src.models.factory import ModelFactory  # noqa: E402


@pytest.mark.unit
class TestEncoding:
    def test_alphabet_sizes(self) -> None:
        assert len(ALPHABET_LOWER) == 70
        assert len(ALPHABET_MIXED) == 96

    def test_lowercase_unknown_and_reverse(self) -> None:
        # "A bé": lowercased, space and unknown char -> 0; padded then reversed
        row = encode_texts(["Ab c"], ALPHABET_LOWER, True, max_length=6)[0]
        # pre-padded: [0,0,1,2,0,3] -> reversed: [3,0,2,1,0,0]
        assert row.tolist() == [3, 0, 2, 1, 0, 0]

    def test_mixed_case_keeps_case(self) -> None:
        row = encode_texts(["aA"], ALPHABET_MIXED, False, max_length=3)[0]
        assert row.tolist() == [27, 1, 0]
        lowered = encode_texts(["aA"], ALPHABET_LOWER, True, max_length=3)[0]
        assert lowered.tolist() == [1, 1, 0]

    def test_pre_truncation_keeps_last_chars(self) -> None:
        row = encode_texts(["abcdef"], ALPHABET_LOWER, True, max_length=3)[0]
        assert row.tolist() == [6, 5, 4]  # f e d reversed order of "def"

    def test_em_dash_and_newline_indices(self) -> None:
        row = encode_texts(["—\n"], ALPHABET_LOWER, True, max_length=2)[0]
        assert row.tolist() == [70, 37]


@pytest.mark.unit
class TestCharCNN:
    @pytest.mark.parametrize(
        "alphabet,params",
        [(ALPHABET_LOWER, 11_345_775), (ALPHABET_MIXED, 11_396_709)],
    )
    def test_parameter_counts_match_notebook(self, alphabet: str, params: int) -> None:
        model = CharCNN(len(alphabet), 5)
        assert model.flatten_size == 8704
        assert sum(p.numel() for p in model.parameters()) == params

    def test_forward_shape_and_frozen_one_hot(self) -> None:
        model = CharCNN(len(ALPHABET_LOWER), 5)
        out = model(torch.zeros(2, 1014, dtype=torch.long))
        assert out.shape == (2, 5)
        assert not model.embedding.weight.requires_grad
        assert model.embedding.weight[0].abs().sum() == 0


@pytest.mark.unit
class TestTrainingLoop:
    def test_split_takes_last_fraction_of_seeded_shuffle(self) -> None:
        train, val = split_indices(10, 0.2, seed=1)
        order = np.random.RandomState(1).permutation(10)
        assert train.tolist() == order[:8].tolist()
        assert val.tolist() == order[8:].tolist()

    def test_split_rejects_tiny_sets(self) -> None:
        with pytest.raises(ValueError):
            split_indices(1, 0.2, seed=0)

    def test_unknown_monitor_rejected(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(monitor="f1")

    def test_early_stopping_restores_best_epoch(self) -> None:
        model = torch.nn.Linear(1, 1)
        stopper = EarlyStopping("val_accuracy", patience=2)
        values = [0.5, 0.7, 0.6, 0.65]  # best at index 1
        stopped_at = None
        for epoch, value in enumerate(values):
            with torch.no_grad():
                model.weight.fill_(float(epoch))
            if stopper.update(epoch, value, model):
                stopped_at = epoch
                break
        assert stopped_at == 3
        stopper.restore(model)
        assert stopper.best_epoch == 1
        assert model.weight.item() == 1.0

    def test_early_stopping_min_mode(self) -> None:
        model = torch.nn.Linear(1, 1)
        stopper = EarlyStopping("val_loss", patience=1)
        assert not stopper.update(0, 1.0, model)
        assert not stopper.update(1, 0.5, model)
        assert stopper.update(2, 0.5, model)  # not a strict improvement

    def test_fit_learns_and_logs(self) -> None:
        torch.manual_seed(0)
        x = torch.randint(0, 2, (100, 4))
        y = (x[:, 0] == 1).long()
        model = torch.nn.Sequential(
            torch.nn.Embedding(2, 8), torch.nn.Flatten(), torch.nn.Linear(32, 2)
        )
        config = TrainingConfig(
            learning_rate=0.05, batch_size=16, max_epochs=15, patience=15,
            reduce_lr_patience=0,
        )
        history = fit(model, x, y, config, seed=0, device=torch.device("cpu"))
        assert 1 <= len(history) <= 15
        assert set(history[0]) >= {"loss", "accuracy", "val_loss", "val_accuracy", "lr"}
        assert max(h["val_accuracy"] for h in history) > 0.9


@pytest.mark.unit
class TestCharCnnStrategy:
    def _strategy(self, cls=CharCnnStrategy):
        return cls(
            ModelConfig(),
            PreprocessingConfig(),
            training_config=TrainingConfig(batch_size=8, max_epochs=2, patience=2),
            max_length=128, filters=8, hidden=16,
        )

    def test_train_predict_tiny(self) -> None:
        texts = [f"great food {i}" if i % 2 else f"awful service {i}" for i in range(40)]
        labels = [i % 2 for i in range(40)]
        strategy = self._strategy()
        strategy.train(texts, labels)
        preds = strategy.predict(texts)
        assert preds.shape == (40,)
        assert set(preds.tolist()) <= {0, 1}
        assert len(strategy.history) >= 1

    def test_predict_before_train_raises(self) -> None:
        with pytest.raises(RuntimeError):
            self._strategy().predict(["x"])

    def test_predict_uses_variant_own_encoder(self, monkeypatch) -> None:
        seen = {}
        real = char_cnn.encode_texts

        def spy(texts, alphabet, lowercase, max_length=1014):
            seen["args"] = (alphabet, lowercase)
            return real(texts, alphabet, lowercase, max_length)

        monkeypatch.setattr(char_cnn, "encode_texts", spy)
        strategy = self._strategy(CharCnnMixedCaseStrategy)
        texts = ["Aa"] * 20
        strategy.train(texts, [0, 1] * 10)
        strategy.predict(texts)
        assert seen["args"] == (ALPHABET_MIXED, False)

    def test_factory_and_experiments(self) -> None:
        for name in ("cnn-char", "cnn-char-mixed-case"):
            assert name in ExperimentConfig.experiment_names()
            config = ExperimentConfig.named(name)
            assert config.requires_extra == "deep"
            assert ModelFactory.create(config.model.model_type, config).get_name()
