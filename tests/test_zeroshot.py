"""Offline tests for the zero-shot experiments (skipped without the deep extra)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.config import ExperimentConfig  # noqa: E402
from src.models.deep import zeroshot  # noqa: E402
from src.models.deep.zeroshot import PIPELINE_DEFAULT_TEMPLATE, ZeroShotStrategy  # noqa: E402
from src.models.factory import ModelFactory  # noqa: E402
from src.pipeline.steps import LoadDataStep, TrainModelStep  # noqa: E402


class FakePipeline:
    """Records calls; ranks the label at ``winners[i]`` first for text i."""

    def __init__(self, winners):
        self.winners = winners
        self.calls = []

    def __call__(self, texts, candidate_labels, **kwargs):
        self.calls.append((list(texts), list(candidate_labels), kwargs))
        out = []
        for text in texts:
            top = candidate_labels[self.winners[text]]
            rest = [c for c in candidate_labels if c != top]
            out.append({"sequence": text, "labels": [top] + rest, "scores": []})
        return out


def build(name, winners=None, **kwargs):
    config = ExperimentConfig.named(name)
    fake = FakePipeline(winners or {})
    strategy = ModelFactory.create("zeroshot", config)
    strategy._classifier = fake
    for key, value in kwargs.items():
        setattr(strategy, key, value)
    return config, strategy, fake


@pytest.mark.unit
class TestZeroShotExperiments:
    @pytest.mark.parametrize(
        "name,labels,template",
        [
            ("zeroshot-form1-val", ["0", "1", "2", "3", "4"], "This review is rated {} out of 4."),
            (
                "zeroshot-form2-val",
                ["very negative", "negative", "neutral", "positive", "very positive"],
                "This review can be considered {}.",
            ),
            (
                "zeroshot-form3-val",
                ["awful", "bad", "neutral", "good", "excellent"],
                "What is being reviewed can be better described as {}",
            ),
            (
                "zeroshot-noverb-val",
                ["very negative", "negative", "neutral", "positive", "very positive"],
                None,
            ),
        ],
    )
    def test_forms(self, name, labels, template) -> None:
        config, strategy, fake = build(name, {"t": 0})
        assert config.requires_extra == "deep"
        assert strategy.candidate_labels == labels
        strategy.predict(["t"])
        _, passed_labels, kwargs = fake.calls[0]
        assert passed_labels == labels
        assert kwargs["multi_label"] is False
        if template is None:  # pipeline default template, never an explicit one
            assert "hypothesis_template" not in kwargs
        else:
            assert kwargs["hypothesis_template"] == template

    def test_default_template_is_the_installed_pipelines(self) -> None:
        import inspect

        from transformers.pipelines.zero_shot_classification import (
            ZeroShotClassificationPipeline,
        )

        default = inspect.signature(ZeroShotClassificationPipeline.preprocess).parameters[
            "hypothesis_template"
        ].default
        assert default == PIPELINE_DEFAULT_TEMPLATE

    def test_validation_experiments_use_notebook_slice_without_train_split(self) -> None:
        for name in ("form1", "form2", "form3", "noverb"):
            data = ExperimentConfig.named(f"zeroshot-{name}-val").data
            assert data.train_split is None
            assert data.test_split == "train[:2%]"
            assert data.task == "stars5"

    def test_test_and_task_experiments(self) -> None:
        test = ExperimentConfig.named("zeroshot-test")
        assert (test.data.test_split, test.data.task) == ("test", "stars5")
        three = ExperimentConfig.named("zeroshot-3class")
        assert three.data.task == "stars3"
        assert three.model.candidate_labels == ["negative", "neutral", "positive"]
        assert three.model.hypothesis_template == "This review can be considered {}."
        binary = ExperimentConfig.named("zeroshot-binary")
        assert binary.data.task == "polarity_no_neutral"
        assert binary.model.candidate_labels == ["negative", "positive"]
        assert binary.data.train_split is None and binary.data.test_split == "test"


@pytest.mark.unit
class TestZeroShotStrategy:
    def test_label_to_id_mapping(self) -> None:
        texts = ["a", "b", "c", "d", "e"]
        _, strategy, _ = build("zeroshot-form2-val", {"a": 4, "b": 0, "c": 2, "d": 3, "e": 1})
        assert strategy.predict(texts).tolist() == [4, 0, 2, 3, 1]

    def test_form1_digit_labels_map_back_to_ints(self) -> None:
        _, strategy, _ = build("zeroshot-form1-val", {"a": 3, "b": 0})
        assert strategy.predict(["a", "b"]).tolist() == [3, 0]

    def test_three_and_two_class_ids(self) -> None:
        _, three, _ = build("zeroshot-3class", {"a": 2, "b": 1})
        assert three.predict(["a", "b"]).tolist() == [2, 1]
        _, binary, _ = build("zeroshot-binary", {"a": 1, "b": 0})
        assert binary.predict(["a", "b"]).tolist() == [1, 0]

    def test_batch_size_passed_and_all_texts_in_one_call(self) -> None:
        _, strategy, fake = build("zeroshot-form2-val", {"a": 0, "b": 1}, batch_size=5)
        strategy.predict(["a", "b"])
        assert len(fake.calls) == 1 and fake.calls[0][2]["batch_size"] == 5

    def test_texts_are_processed_in_chunks_with_same_predictions(self, capsys) -> None:
        winners = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        _, strategy, fake = build("zeroshot-form2-val", winners, chunk_size=2)
        assert strategy.predict(list("abcde")).tolist() == [0, 1, 2, 3, 4]
        assert [c[0] for c in fake.calls] == [["a", "b"], ["c", "d"], ["e"]]
        assert "5/5" in capsys.readouterr().out

    def test_single_dict_result_is_accepted(self) -> None:
        _, strategy, _ = build("zeroshot-binary")
        strategy._classifier = lambda texts, candidate_labels, **kw: {
            "labels": ["positive", "negative"]
        }
        assert strategy.predict(["x"]).tolist() == [1]

    def test_empty_predict(self) -> None:
        _, strategy, fake = build("zeroshot-binary")
        assert strategy.predict([]).tolist() == []
        assert fake.calls == []

    def test_train_is_a_noop(self) -> None:
        _, strategy, fake = build("zeroshot-binary")
        assert strategy.train([], []) is None
        assert fake.calls == []

    def test_label_count_must_match_task(self) -> None:
        config = ExperimentConfig.named("zeroshot-3class")
        config.model.candidate_labels = ["negative", "positive"]
        with pytest.raises(ValueError, match="2 candidate labels"):
            ModelFactory.create("zeroshot", config)

    def test_missing_labels_fail_loudly(self) -> None:
        config = ExperimentConfig.named("lr-bow")
        with pytest.raises(ValueError, match="candidate_labels"):
            ModelFactory.create("zeroshot", config)

    def test_pipeline_built_on_requested_device_without_system_error(self) -> None:
        _, strategy, _ = build("zeroshot-binary")
        strategy._classifier = None
        with patch.object(zeroshot, "pipeline") as make_pipeline:
            strategy._pipeline()
        args, kwargs = make_pipeline.call_args
        assert args == ("zero-shot-classification",)
        assert kwargs["model"] == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
        assert kwargs["device"].type in ("cuda", "cpu")


@pytest.mark.unit
class TestZeroShotPipelineSteps:
    @patch("src.pipeline.steps.DataLoader")
    def test_no_train_split_is_loaded(self, mock_loader_class) -> None:
        loader = MagicMock()
        loader.load_test.return_value = (["t"], [1])
        loader.class_names = ["negative", "positive"]
        mock_loader_class.return_value = loader
        context = LoadDataStep(ExperimentConfig.named("zeroshot-binary")).execute({})
        loader.load_train.assert_not_called()
        assert context["train_texts"] == [] and context["test_texts"] == ["t"]

    def test_train_step_skips_class_check_and_predicts(self) -> None:
        config = ExperimentConfig.named("zeroshot-binary")
        fake = FakePipeline({"t": 1})
        with patch.object(ZeroShotStrategy, "_pipeline", return_value=fake):
            context = TrainModelStep(config).execute(
                {"train_texts": [], "train_labels": [], "test_texts": ["t"]}
            )
        assert context["predictions"].tolist() == [1]
        assert isinstance(context["trained_strategy"], ZeroShotStrategy)
        assert np.issubdtype(context["predictions"].dtype, np.integer)
