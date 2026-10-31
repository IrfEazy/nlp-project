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
from src.models.deep import lstm_glove  # noqa: E402
from src.models.deep.lstm_glove import (  # noqa: E402
    BiLstmClassifier,
    LstmGloveStrategy,
    build_embedding_matrix,
    build_vocabulary,
    encode_texts as encode_words,
    notebook_training_config,
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
            reduce_lr_patience=1,
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
            n_classes=2,
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

    def test_class_count_comes_from_the_task_not_the_labels(self) -> None:
        config = ExperimentConfig.named("cnn-char")
        config.data.task = "stars3"
        strategy = ModelFactory.create("cnn-char", config)
        assert strategy.n_classes == 3
        strategy.training_config = TrainingConfig(batch_size=8, max_epochs=1)
        strategy.max_length, strategy.filters, strategy.hidden = 128, 8, 16
        strategy.train(["good", "bad"] * 10, [0, 1] * 10)  # labels never reach 2
        assert strategy.model.classifier[-1].out_features == 3


@pytest.mark.unit
class TestLstmVocabularyAndEncoding:
    def test_threshold_and_first_occurrence_order(self) -> None:
        texts = ["b a rare", "a b a", "c b"]
        vocab = build_vocabulary(texts, min_frequency=2)
        assert vocab == {"b": 1, "a": 2}  # counts b=3 a=3 c=1 rare=1, order of first use

    def test_oov_is_zero_in_place_and_post_padding(self) -> None:
        vocab = {"good": 1, "food": 2}
        row = encode_words(["good unknown food"], vocab, max_length=5)[0]
        assert row.tolist() == [1, 0, 2, 0, 0]

    def test_post_truncation_keeps_first_words(self) -> None:
        vocab = {"a": 1, "b": 2, "c": 3}
        assert encode_words(["a b c a b"], vocab, max_length=3)[0].tolist() == [1, 2, 3]

    def test_keras_filter_chars_become_separators_when_encoding(self) -> None:
        # the vocabulary keeps the backslash token (split), encoding filters like Keras
        vocab = build_vocabulary(["\\nworst \\nworst"], min_frequency=2)
        assert vocab == {"\\nworst": 1}
        assert encode_words(["\\nworst"], vocab, max_length=2)[0].tolist() == [0, 0]
        assert encode_words(["\\nworst"], {"nworst": 1}, max_length=2)[0].tolist() == [1, 0]
        # apostrophes are kept, underscores are filtered, case is folded
        row = encode_words(["Don't a_b"], {"don't": 1, "a": 2, "b": 3}, 4)[0]
        assert row.tolist() == [1, 2, 3, 0]

    def test_embedding_matrix_from_fake_glove(self) -> None:
        vocab = {"good": 1, "zzz": 2, "food": 3}
        vectors = {"good": np.arange(4, dtype=float), "food": np.ones(4)}
        matrix = build_embedding_matrix(vocab, vectors, dim=4)
        assert matrix.shape == (4, 4)
        assert matrix[0].tolist() == [0, 0, 0, 0]
        assert matrix[1].tolist() == [0, 1, 2, 3]
        assert matrix[2].tolist() == [0, 0, 0, 0]  # not in GloVe
        assert matrix[3].tolist() == [1, 1, 1, 1]


@pytest.mark.unit
class TestBiLstmClassifier:
    def test_parameter_count_and_frozen_embedding(self) -> None:
        model = BiLstmClassifier(torch.zeros(23743, 300), 5)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Keras: 1,568,261; PyTorch keeps two LSTM bias vectors per direction (+2048)
        assert trainable == 1_570_309
        assert sum(p.numel() for p in model.parameters()) == 7_122_900 + 1_570_309
        assert not model.embedding.weight.requires_grad

    def test_forward_shape_and_keras_like_init(self) -> None:
        model = BiLstmClassifier(torch.randn(11, 6), 3, hidden_size=4)
        assert model(torch.randint(0, 11, (2, 9))).shape == (2, 3)
        for name, param in model.lstm.named_parameters():
            if name.startswith("bias_ih"):
                assert param[4:8].eq(1).all() and param[:4].eq(0).all()
            elif name.startswith("bias_hh"):
                assert param.eq(0).all()
        w = model.lstm.weight_hh_l0
        assert torch.allclose(w.T @ w, torch.eye(4), atol=1e-5)  # orthonormal columns

    def test_head_input_is_final_state_of_both_directions(self) -> None:
        torch.manual_seed(0)
        model = BiLstmClassifier(torch.randn(5, 3), 2, hidden_size=4).eval()
        x = torch.tensor([[1, 2, 3, 4]])
        out, (h_n, _) = model.lstm(model.embedding(x))
        # forward: last step; backward: its output at step 0 (it read the sequence from the end)
        assert torch.allclose(h_n[0], out[:, -1, :4])
        assert torch.allclose(h_n[1], out[:, 0, 4:])
        assert torch.allclose(model(x), model.head(torch.cat([h_n[0], h_n[1]], dim=1)))


def _tiny_model():
    return torch.nn.Sequential(
        torch.nn.Embedding(2, 3), torch.nn.Flatten(), torch.nn.Linear(6, 2)
    )


@pytest.mark.unit
class TestKerasTrainingSemantics:
    def test_no_shuffle_split_is_dataset_order(self) -> None:
        train, val = split_indices(10, 0.2, seed=3, shuffle=False)
        assert train.tolist() == list(range(8)) and val.tolist() == [8, 9]

    def test_notebook_config(self) -> None:
        c = notebook_training_config()
        assert (c.learning_rate, c.batch_size, c.max_epochs, c.monitor, c.patience) == (
            1e-3, 32, 10, "val_loss", 1,
        )
        assert c.restore_best_weights and not c.shuffle_before_split
        assert (c.reduce_lr_patience, c.reduce_lr_factor, c.reduce_lr_min_delta) == (1, 0.1, 1e-4)

    def test_best_val_loss_weights_restored(self, monkeypatch) -> None:
        losses = iter([0.9, 0.5, 0.7, 0.6])
        monkeypatch.setattr(
            "src.models.deep.training._evaluate", lambda *a, **k: (next(losses), 0.5)
        )
        model = _tiny_model()
        x, y = torch.randint(0, 2, (20, 2)), torch.randint(0, 2, (20,))
        snapshots = []
        orig = EarlyStopping.update

        def spy(self, epoch, value, mdl):
            snapshots.append(mdl[2].weight.detach().clone())
            return orig(self, epoch, value, mdl)

        monkeypatch.setattr(EarlyStopping, "update", spy)
        history = fit(
            model, x, y,
            TrainingConfig(batch_size=4, max_epochs=10, monitor="val_loss", patience=1,
                           learning_rate=0.1),
            seed=0, device=torch.device("cpu"),
        )
        assert len(history) == 3  # epoch 3 is worse than epoch 2 -> stop
        assert torch.equal(model[2].weight.detach(), snapshots[1])  # best epoch (0.5)

    def test_reduce_lr_keras_patience_and_absolute_min_delta(self, monkeypatch) -> None:
        # Keras: patience 1 reduces after ONE epoch without improvement beyond 1e-4.
        losses = iter([1.0, 0.99995, 0.5, 0.5, 0.5])
        monkeypatch.setattr(
            "src.models.deep.training._evaluate", lambda *a, **k: (next(losses), 0.5)
        )
        x, y = torch.randint(0, 2, (20, 2)), torch.randint(0, 2, (20,))
        history = fit(
            _tiny_model(), x, y,
            TrainingConfig(batch_size=20, max_epochs=5, monitor="val_loss", patience=9,
                           reduce_lr_patience=1, learning_rate=1.0),
            seed=0, device=torch.device("cpu"),
        )
        lrs = [h["lr"] for h in history]
        # epoch 2 (0.99995) improves < 1e-4 -> reduced before epoch 3; epoch 4 stalls -> again
        assert lrs == pytest.approx([1.0, 1.0, 0.1, 0.1, 0.01])

    def test_reduce_lr_patience_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(reduce_lr_patience=0)


@pytest.mark.unit
class TestLstmGloveStrategy:
    def _strategy(self, **kwargs):
        vectors = {w: np.random.RandomState(i).randn(300) for i, w in enumerate(
            ["good", "food", "bad", "service"])}
        return LstmGloveStrategy(
            ModelConfig(),
            PreprocessingConfig(remove_stopwords=False, expand_contractions=False),
            n_classes=kwargs.pop("n_classes", 3),
            training_config=TrainingConfig(batch_size=8, max_epochs=2, patience=2,
                                           monitor="val_loss"),
            max_length=6, min_word_frequency=2, hidden_size=8, vectors=vectors,
            **kwargs,
        )

    @pytest.fixture(autouse=True)
    def _no_nltk(self, stub_nltk):
        """Offline tokenization and stopwords (conftest)."""

    def test_train_predict_tiny_and_task_class_count(self) -> None:
        texts = ["good food", "bad service", "good service"] * 10
        labels = [0, 1, 0] * 10  # label 2 never occurs but the task has 3 classes
        strategy = self._strategy()
        strategy.train(texts, labels)
        assert set(strategy.vocabulary) == {"good", "food", "bad", "service"}
        assert strategy.model.head[-1].out_features == 3
        assert strategy.predict(texts).shape == (30,)
        assert len(strategy.history) >= 1

    def test_requires_class_count_and_training(self) -> None:
        strategy = self._strategy()
        with pytest.raises(RuntimeError):
            strategy.predict(["x"])
        strategy.n_classes = None
        with pytest.raises(ValueError):
            strategy.train(["a"], [0])

    def test_preprocessor_is_used(self, monkeypatch) -> None:
        calls = []
        real = lstm_glove.TextPreprocessor.preprocess_batch

        def spy(self, texts):
            calls.append(len(texts))
            return real(self, texts)

        monkeypatch.setattr(lstm_glove.TextPreprocessor, "preprocess_batch", spy)
        strategy = self._strategy()
        strategy.train(["Good Food!"] * 20, [0, 1] * 10)
        strategy.predict(["Good"])
        assert calls == [20, 1]

    def test_experiments_registered(self) -> None:
        five = ExperimentConfig.named("lstm-glove")
        three = ExperimentConfig.named("lstm-glove-3class")
        assert (five.data.task, three.data.task) == ("stars5", "stars3")
        assert five.requires_extra == three.requires_extra == "deep"
        assert ModelFactory.create("lstm-glove", three).n_classes == 3
