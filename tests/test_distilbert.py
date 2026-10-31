"""Offline tests for the DistilBERT strategies (skipped without the deep extra)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from sklearn.model_selection import train_test_split  # noqa: E402
from transformers import DistilBertConfig, DistilBertForSequenceClassification  # noqa: E402
from transformers import BertTokenizer  # noqa: E402

from src.config import ExperimentConfig  # noqa: E402
from src.models.deep import distilbert  # noqa: E402
from src.models.deep.distilbert import (  # noqa: E402
    DistilBertFineTuneStrategy,
    DistilBertPolarityStrategy,
    DistilBertTransferStrategy,
    TokenizedTexts,
    TransformerClassifier,
    freeze_encoder,
    parameter_summary,
)
from src.models.deep.training import TrainingConfig, fit  # noqa: E402
from src.models.factory import ModelFactory  # noqa: E402

WORDS = ["good", "bad", "food", "service", "great", "awful", "the", "was", "very", "nice"]
TEXTS = [
    "the food was very good", "awful service", "great nice food", "bad bad service",
    "very nice", "the service was awful and bad", "good good great", "bad food",
] * 5
LABELS = [1, 0, 1, 0, 1, 0, 1, 0] * 5


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS + ["and"]
    path.write_text("\n".join(tokens) + "\n")
    return BertTokenizer(str(path))


def tiny_hf_model(n_labels: int = 2):
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=16, dim=16, hidden_dim=32, n_layers=1, n_heads=2,
        max_position_embeddings=64, num_labels=n_labels,
    )
    return DistilBertForSequenceClassification(config)


def make(cls, tokenizer, n_labels=2, **overrides):
    config = cls.default_training_config()
    for key, value in {"batch_size": 8, "learning_rate": 1e-2, **overrides}.items():
        setattr(config, key, value)
    experiment = ExperimentConfig.named(
        "distilbert-polarity" if cls is DistilBertPolarityStrategy else "distilbert-finetune"
    )
    return cls(
        experiment.model, experiment.preprocessing, n_classes=n_labels,
        training_config=config, tokenizer=tokenizer, model=tiny_hf_model(n_labels),
        max_length=16,
    )


def snapshot(module):
    return {k: v.clone() for k, v in module.named_parameters()}


@pytest.mark.unit
class TestTokenizedTexts:
    def test_max_length_padding_is_fixed(self, tokenizer) -> None:
        inputs = TokenizedTexts(tokenizer, ["good", "good food was"], "max_length", max_length=12)
        batch = inputs[torch.tensor([0, 1])]
        assert batch["input_ids"].shape == (2, 12)
        assert batch["attention_mask"].sum(dim=1).tolist() == [3, 5]  # CLS + words + SEP

    def test_dynamic_padding_follows_batch(self, tokenizer) -> None:
        inputs = TokenizedTexts(tokenizer, ["good", "good food was", "bad"], "dynamic")
        assert inputs[torch.tensor([0, 2])]["input_ids"].shape == (2, 3)
        assert inputs[torch.tensor([0, 1])]["input_ids"].shape == (2, 5)

    def test_truncation(self, tokenizer) -> None:
        inputs = TokenizedTexts(tokenizer, ["good " * 30], "dynamic", max_length=8)
        assert inputs[torch.tensor([0])]["input_ids"].shape == (1, 8)

    def test_unknown_padding_rejected(self, tokenizer) -> None:
        with pytest.raises(ValueError):
            TokenizedTexts(tokenizer, ["good"], "left")


@pytest.mark.unit
class TestParametersAndSchedule:
    def test_summary_counts_frozen_and_trainable(self) -> None:
        model = TransformerClassifier(tiny_hf_model())
        total = sum(p.numel() for p in model.parameters())
        assert parameter_summary(model) == (total, 0)
        freeze_encoder(model)
        trainable, frozen = parameter_summary(model)
        head = sum(
            p.numel()
            for n, p in model.model.named_parameters()
            if n.startswith(("pre_classifier", "classifier"))
        )
        assert trainable == head
        assert frozen == total - head
        assert all(not p.requires_grad for p in model.model.distilbert.parameters())

    def test_linear_decay_reaches_zero_and_steps_every_iteration(self, tokenizer) -> None:
        strategy = make(DistilBertFineTuneStrategy, tokenizer, evaluate_train=False)
        strategy.model = strategy._build_model()
        lrs = []
        original = torch.optim.AdamW.step

        def spy(self_, *args, **kwargs):
            lrs.append(self_.param_groups[0]["lr"])
            return original(self_, *args, **kwargs)

        torch.optim.AdamW.step = spy
        try:
            history = strategy._fit(TEXTS, torch.tensor(LABELS), 42)
        finally:
            torch.optim.AdamW.step = original
        steps = len(lrs)
        assert steps == -(-30 // 8)  # 75% of 40 samples, batch 8
        assert lrs[0] == pytest.approx(1e-2)
        assert lrs == sorted(lrs, reverse=True) and len(set(lrs)) == steps
        assert history[-1]["lr"] == 0.0

    def test_weight_decay_skips_biases_and_norms(self, tokenizer) -> None:
        from src.models.deep.training import _build_optimizer

        model = TransformerClassifier(tiny_hf_model())
        optimizer = _build_optimizer(
            model, TrainingConfig(optimizer="adamw", weight_decay=0.01)
        )
        decays = {g["weight_decay"]: g["params"] for g in optimizer.param_groups}
        assert all(p.ndim >= 2 for p in decays[0.01])
        assert all(p.ndim < 2 for p in decays[0.0])

    def test_unknown_optimizer_rejected(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(optimizer="sgd")


@pytest.mark.unit
class TestFineTuneAndTransfer:
    def test_finetune_updates_encoder_params(self, tokenizer) -> None:
        strategy = make(DistilBertFineTuneStrategy, tokenizer)
        hf_model = strategy._hf_model
        before = snapshot(hf_model.distilbert)
        strategy.train(TEXTS, LABELS)
        after = snapshot(hf_model.distilbert)
        assert any(not torch.equal(before[k], after[k]) for k in before)

    def test_transfer_freezes_encoder_and_trains_head(self, tokenizer, capsys) -> None:
        strategy = make(DistilBertTransferStrategy, tokenizer)
        hf_model = strategy._hf_model
        encoder_before, head_before = snapshot(hf_model.distilbert), snapshot(hf_model.classifier)
        strategy.train(TEXTS, LABELS)
        encoder_after, head_after = snapshot(hf_model.distilbert), snapshot(hf_model.classifier)
        assert all(torch.equal(encoder_before[k], encoder_after[k]) for k in encoder_before)
        assert any(not torch.equal(head_before[k], head_after[k]) for k in head_before)
        trainable, frozen = parameter_summary(strategy.model)
        assert f"Trainable parameters: {trainable:,}; frozen parameters: {frozen:,}" in (
            capsys.readouterr().out
        )

    def test_validation_split_is_seeded_75_25(self, tokenizer) -> None:
        strategy = make(DistilBertFineTuneStrategy, tokenizer, evaluate_train=True)
        strategy.train(TEXTS, LABELS)
        assert strategy.training_config.validation_split == 0.25
        assert strategy.training_config.shuffle_before_split
        from src.models.deep.training import split_indices

        train, val = split_indices(len(TEXTS), 0.25, 42)
        assert (len(train), len(val)) == (30, 10)

    def test_predictions_are_canonical_label_ids(self, tokenizer) -> None:
        strategy = make(DistilBertFineTuneStrategy, tokenizer, n_labels=3)
        strategy.train(TEXTS, LABELS)
        predictions = strategy.predict(TEXTS[:10])
        assert predictions.shape == (10,)
        assert set(predictions.tolist()) <= {0, 1, 2}

    def test_predict_before_train_raises(self, tokenizer) -> None:
        with pytest.raises(RuntimeError):
            make(DistilBertFineTuneStrategy, tokenizer).predict(["good"])

    def test_requires_n_classes(self, tokenizer) -> None:
        strategy = make(DistilBertFineTuneStrategy, tokenizer)
        strategy.n_classes = None
        with pytest.raises(ValueError, match="n_classes"):
            strategy.train(TEXTS, LABELS)


@pytest.mark.unit
class TestPolarity:
    def test_config_matches_notebook(self) -> None:
        config = DistilBertPolarityStrategy.default_training_config()
        assert (config.batch_size, config.learning_rate, config.weight_decay) == (32, 2e-5, 0.01)
        assert config.max_epochs == 1 and config.linear_decay
        assert DistilBertPolarityStrategy.split_seed == 2307
        assert DistilBertPolarityStrategy.validation_fraction == 0.2
        assert DistilBertPolarityStrategy.padding == "dynamic"

    def test_split_follows_sklearn_semantics(self, tokenizer, monkeypatch) -> None:
        seen = {}

        def fake_fit(model, inputs, labels, config, seed, validation=None):
            seen.update(train=len(labels), val=len(validation[1]), labels=labels.tolist(),
                        val_labels=validation[1].tolist())
            return [{"val_accuracy": 0.5}]

        monkeypatch.setattr(distilbert, "fit", fake_fit)
        strategy = make(DistilBertPolarityStrategy, tokenizer)
        strategy.train(TEXTS, LABELS)
        _, _, expected_train, expected_val = train_test_split(
            TEXTS, LABELS, test_size=0.2, random_state=2307
        )
        assert (seen["train"], seen["val"]) == (32, 8)
        assert seen["labels"] == expected_train and seen["val_labels"] == expected_val

    def test_punctuation_is_deleted_before_tokenizing(self, tokenizer) -> None:
        strategy = make(DistilBertPolarityStrategy, tokenizer)
        assert strategy._clean(["good, food!"]) == ["good food"]

    def test_trains_and_predicts_with_dynamic_padding(self, tokenizer) -> None:
        strategy = make(DistilBertPolarityStrategy, tokenizer)
        strategy.train(TEXTS, LABELS)
        assert len(strategy.history) == 1
        assert strategy.predict(TEXTS[:5]).shape == (5,)


@pytest.mark.unit
class TestFitExtensions:
    def test_validation_argument_overrides_split(self) -> None:
        x = torch.randint(0, 2, (20, 4))
        y = (x[:, 0] == 1).long()
        model = torch.nn.Sequential(
            torch.nn.Embedding(2, 4), torch.nn.Flatten(), torch.nn.Linear(16, 2)
        )
        history = fit(
            model, x, y, TrainingConfig(batch_size=5, max_epochs=1), seed=0,
            device=torch.device("cpu"), validation=(x[:4], y[:4]),
        )
        assert len(history) == 1

    def test_grad_clipping_runs(self) -> None:
        x = torch.randint(0, 2, (20, 4))
        y = (x[:, 0] == 1).long()
        model = torch.nn.Sequential(
            torch.nn.Embedding(2, 4), torch.nn.Flatten(), torch.nn.Linear(16, 2)
        )
        config = TrainingConfig(batch_size=5, max_epochs=2, max_grad_norm=1.0,
                                optimizer="adamw", weight_decay=0.01, linear_decay=True)
        history = fit(model, x, y, config, seed=0, device=torch.device("cpu"))
        assert history[-1]["lr"] == 0.0


@pytest.mark.unit
class TestRegistration:
    @pytest.mark.parametrize("name", ["distilbert-finetune", "distilbert-transfer", "distilbert-polarity"])
    def test_factory_builds_strategies_with_task_classes(self, name) -> None:
        config = ExperimentConfig.named(name)
        strategy = ModelFactory.create(config.model.model_type, config)
        assert strategy.n_classes == (2 if name == "distilbert-polarity" else 5)


def test_chunked_tokenization_matches_single_call(tokenizer):
    whole = TokenizedTexts(tokenizer, TEXTS, "dynamic", chunk_size=10_000)
    chunked = TokenizedTexts(tokenizer, TEXTS, "dynamic", chunk_size=3)
    assert len(chunked) == len(whole) == len(TEXTS)
    expected = tokenizer(TEXTS, truncation=True, max_length=512)["input_ids"]
    assert [r.tolist() for r in chunked.ids] == expected
    index = torch.arange(len(TEXTS))
    for key in ("input_ids", "attention_mask"):
        assert torch.equal(chunked[index][key], whole[index][key])
