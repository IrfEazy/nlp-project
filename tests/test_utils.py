"""
Tests for the utils module.

Tests cover:
- configure_environment function
- Environment configuration for reproducibility
"""

import os
import random

import numpy as np
import pytest

from src.utils.reproducibility import configure_environment, get_seed


@pytest.mark.unit
class TestConfigureEnvironment:
    """Tests for configure_environment function."""

    def test_sets_random_seed(self) -> None:
        """configure_environment should set Python random seed."""
        configure_environment(seed=123)

        # After seeding, random should be deterministic
        random.seed(123)
        expected = random.random()

        configure_environment(seed=123)
        actual = random.random()

        assert actual == expected

    def test_sets_numpy_seed(self) -> None:
        """configure_environment should set NumPy random seed."""
        configure_environment(seed=456)

        # After seeding, numpy random should be deterministic
        np.random.seed(456)
        expected = np.random.rand()

        configure_environment(seed=456)
        actual = np.random.rand()

        assert actual == expected

    def test_sets_python_hash_seed(self) -> None:
        """configure_environment should set PYTHONHASHSEED env var."""
        configure_environment(seed=789)

        assert os.environ["PYTHONHASHSEED"] == "789"

    def test_sets_tf_logging_level(self) -> None:
        """configure_environment should set TF_CPP_MIN_LOG_LEVEL."""
        configure_environment(seed=42)

        assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "3"

    def test_sets_mpl_config_dir(self) -> None:
        """configure_environment should set MPLCONFIGDIR."""
        configure_environment(seed=42)

        assert "MPLCONFIGDIR" in os.environ
        assert "configs" in os.environ["MPLCONFIGDIR"]

    def test_default_seed_value(self) -> None:
        """configure_environment should use seed 42 by default."""
        configure_environment()

        assert get_seed() == 42

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds should produce different random sequences."""
        configure_environment(seed=100)
        result1 = np.random.rand()

        configure_environment(seed=200)
        result2 = np.random.rand()

        assert result1 != result2

    def test_same_seed_produces_reproducible_results(self) -> None:
        """Same seed should produce identical random sequences."""
        configure_environment(seed=555)
        sequence1 = [np.random.rand() for _ in range(5)]

        configure_environment(seed=555)
        sequence2 = [np.random.rand() for _ in range(5)]

        assert sequence1 == sequence2


@pytest.mark.unit
class TestGetSeed:
    """Tests for get_seed function."""

    def test_returns_configured_seed(self) -> None:
        """get_seed should return the seed set by configure_environment."""
        configure_environment(seed=999)

        assert get_seed() == 999

    def test_returns_different_seed_after_reconfigure(self) -> None:
        """get_seed should reflect the latest configured seed."""
        configure_environment(seed=111)
        assert get_seed() == 111

        configure_environment(seed=222)
        assert get_seed() == 222
        assert get_seed() == 222
