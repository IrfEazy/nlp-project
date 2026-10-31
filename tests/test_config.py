"""
Tests for the config module.

Tests cover:
- All configuration dataclasses
- Factory methods for common configurations
"""

import pytest

from src.config import DataConfig, ExperimentConfig, ModelConfig, PreprocessingConfig


@pytest.mark.unit
class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self) -> None:
        """DataConfig should have sensible defaults."""
        config = DataConfig()

        assert config.dataset_name == "Yelp/yelp_review_full"
        assert config.train_split == "train"
        assert config.test_split == "test"
        assert config.sample_size is None

    def test_custom_values(self) -> None:
        """DataConfig should accept custom values."""
        config = DataConfig(
            dataset_name="custom_dataset",
            train_split="training",
            test_split="validation",
            sample_size=1000,
        )

        assert config.dataset_name == "custom_dataset"
        assert config.train_split == "training"
        assert config.test_split == "validation"
        assert config.sample_size == 1000


@pytest.mark.unit
class TestPreprocessingConfig:
    """Tests for PreprocessingConfig dataclass."""

    def test_default_values(self) -> None:
        """PreprocessingConfig should have sensible defaults."""
        config = PreprocessingConfig()

        assert config.min_df == 50
        assert config.max_df == 0.5
        assert config.remove_stopwords
        assert config.expand_contractions
        assert config.lowercase

    def test_custom_values(self) -> None:
        """PreprocessingConfig should accept custom values."""
        config = PreprocessingConfig(
            min_df=100,
            max_df=0.8,
            remove_stopwords=False,
            expand_contractions=False,
            lowercase=False,
        )

        assert config.min_df == 100
        assert config.max_df == 0.8
        assert not config.remove_stopwords
        assert not config.expand_contractions
        assert not config.lowercase


@pytest.mark.unit
class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self) -> None:
        """ModelConfig should have sensible defaults."""
        config = ModelConfig()

        assert config.model_type == "bow"
        assert config.random_seed == 42
        assert config.max_iter == 100
        assert config.class_weight is None

    def test_custom_values(self) -> None:
        """ModelConfig should accept custom values."""
        config = ModelConfig(
            model_type="tfidf", random_seed=123, max_iter=500, class_weight="balanced"
        )

        assert config.model_type == "tfidf"
        assert config.random_seed == 123
        assert config.max_iter == 500
        assert config.class_weight == "balanced"


@pytest.mark.unit
class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_values(self) -> None:
        """ExperimentConfig should have default nested configs."""
        config = ExperimentConfig()

        assert isinstance(config.data, DataConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.model, ModelConfig)
        assert config.experiment_name == "default_experiment"

    def test_custom_nested_configs(self) -> None:
        """ExperimentConfig should accept custom nested configs."""
        data_config = DataConfig(sample_size=500)
        preprocessing_config = PreprocessingConfig(min_df=25)
        model_config = ModelConfig(model_type="tfidf")

        config = ExperimentConfig(
            data=data_config,
            preprocessing=preprocessing_config,
            model=model_config,
            experiment_name="custom_experiment",
        )

        assert config.data.sample_size == 500
        assert config.preprocessing.min_df == 25
        assert config.model.model_type == "tfidf"
        assert config.experiment_name == "custom_experiment"

    @pytest.mark.parametrize(
        "factory_method,expected_model_type,expected_experiment_name",
        [
            ("for_bow_baseline", "bow", "bow_baseline"),
            ("for_tfidf", "tfidf", "tfidf_logistic"),
        ],
        ids=["bow-factory", "tfidf-factory"],
    )
    def test_factory_methods_create_correct_configs(
        self, factory_method, expected_model_type, expected_experiment_name
    ) -> None:
        """Factory methods should create configurations with correct model type."""
        factory = getattr(ExperimentConfig, factory_method)
        config = factory()

        assert config.model.model_type == expected_model_type
        assert config.experiment_name == expected_experiment_name

    @pytest.mark.parametrize(
        "factory_method,expected_remove_stopwords",
        [
            ("for_bow_baseline", True),
            ("for_tfidf", True),
        ],
        ids=["bow-removes-stopwords", "tfidf-removes-stopwords"],
    )
    def test_factory_methods_configure_stopword_removal(
        self, factory_method, expected_remove_stopwords
    ) -> None:
        """Factory methods should configure stopword removal appropriately."""
        factory = getattr(ExperimentConfig, factory_method)
        config = factory()

        assert config.preprocessing.remove_stopwords == expected_remove_stopwords

    @pytest.mark.parametrize(
        "model_type,expected_name",
        [("bow", "bow_baseline"), ("tfidf", "tfidf_logistic")],
    )
    def test_for_model_returns_preset(self, model_type, expected_name) -> None:
        """for_model should return the preset for a known model type."""
        config = ExperimentConfig.for_model(model_type)

        assert config.model.model_type == model_type
        assert config.experiment_name == expected_name

    def test_for_model_unknown_raises(self) -> None:
        """for_model should raise ValueError naming available presets."""
        with pytest.raises(ValueError, match="bow.*tfidf"):
            ExperimentConfig.for_model("distilbert")

    def test_factory_methods_return_new_instances(self) -> None:
        """Factory methods should return independent instances."""
        config1 = ExperimentConfig.for_bow_baseline()
        config2 = ExperimentConfig.for_bow_baseline()

        # Should be different objects
        assert config1 is not config2

        # Modifying one should not affect the other
        config1.experiment_name = "modified"
        assert config2.experiment_name == "bow_baseline"


@pytest.mark.unit
class TestConfigIntegration:
    """Integration tests for configuration usage."""

    def test_experiment_config_can_be_modified(self) -> None:
        """ExperimentConfig should be modifiable after creation."""
        config = ExperimentConfig.for_bow_baseline()

        # Should be able to modify data config
        config.data = DataConfig(sample_size=1000)

        assert config.data.sample_size == 1000
        # Other configs should remain unchanged
        assert config.model.model_type == "bow"

    def test_config_values_accessible_through_experiment(self) -> None:
        """Nested config values should be accessible."""
        config = ExperimentConfig.for_tfidf()

        # Should be able to access nested values
        assert config.data.dataset_name == "Yelp/yelp_review_full"
        assert config.preprocessing.min_df == 50
        assert config.model.random_seed == 42


@pytest.mark.unit
class TestExperimentRegistry:
    """Tests for the named experiment registry."""

    def test_names(self) -> None:
        assert ExperimentConfig.experiment_names() == [
            "lr-bow",
            "lr-tfidf",
            "lr-bow-3class",
            "lr-bow-polarity",
            "lr-bow-no-neutral",
            "lr-bow-dbpedia",
            "cnn-char",
            "cnn-char-mixed-case",
        ]

    def test_every_name_builds_with_matching_experiment_name(self) -> None:
        for name in ExperimentConfig.experiment_names():
            assert ExperimentConfig.named(name).experiment_name == name

    def test_named_returns_fresh_configs(self) -> None:
        first = ExperimentConfig.named("lr-bow-3class")
        first.data.sample_size = 5
        assert ExperimentConfig.named("lr-bow-3class").data.sample_size is None

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="lr-bow"):
            ExperimentConfig.named("nope")

    def test_vectorizer_params_as_notebook(self) -> None:
        config = ExperimentConfig.named("lr-tfidf")
        assert config.preprocessing.min_df == 50
        assert config.preprocessing.max_df == 0.5
        assert config.preprocessing.remove_stopwords
        assert config.model.model_type == "tfidf"
        assert config.model.max_iter == 100

    def test_three_class(self) -> None:
        config = ExperimentConfig.named("lr-bow-3class")
        assert config.data.task == "stars3"
        assert config.model.class_weight == "balanced"
        assert not config.preprocessing.strip_punctuation

    def test_polarity(self) -> None:
        config = ExperimentConfig.named("lr-bow-polarity")
        assert config.data.task == "polarity"
        assert config.model.class_weight == "balanced"
        assert config.preprocessing.strip_punctuation

    def test_no_neutral_uses_defaults(self) -> None:
        config = ExperimentConfig.named("lr-bow-no-neutral")
        assert config.data.task == "polarity_no_neutral"
        assert config.model.class_weight is None
        assert not config.preprocessing.strip_punctuation

    def test_dbpedia(self) -> None:
        config = ExperimentConfig.named("lr-bow-dbpedia")
        assert config.data.dataset_name == "fancyzhx/dbpedia_14"
        assert config.data.text_field == "content"
        assert config.data.task == "dbpedia14"

    def test_default_experiments_use_raw_text(self) -> None:
        for name in ("lr-bow", "lr-tfidf"):
            config = ExperimentConfig.named(name)
            assert config.data.task == "stars5"
            assert not config.preprocessing.strip_punctuation
