"""Tests for configuration system (config.py).

Tests the three-tier configuration profiles and the load_config() function
with override support.

Tier: 1 (MVP)
Coverage: Profile loading, overrides, validation
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "specHO"))

from config import (
    SpecHOConfig,
    PROFILES,
    load_config,
    ClauseDetectionConfig,
    PairRulesConfig,
    ScoringConfig,
)


class TestProfiles:
    """Tests for predefined configuration profiles."""

    def test_all_profiles_exist(self):
        """Test that all three profiles are defined."""
        assert "simple" in PROFILES
        assert "robust" in PROFILES
        assert "research" in PROFILES

    def test_simple_profile_tier_1(self):
        """Test simple profile has Tier 1 settings."""
        simple = PROFILES["simple"]
        assert simple.tier == 1
        assert simple.profile_name == "simple"
        assert simple.phonetic_analysis.algorithm == "levenshtein"
        assert simple.scoring.aggregation_strategy == "mean"
        assert simple.semantic_analysis.model == "static"

    def test_robust_profile_tier_2(self):
        """Test robust profile has Tier 2 settings."""
        robust = PROFILES["robust"]
        assert robust.tier == 2
        assert robust.profile_name == "robust"
        assert robust.phonetic_analysis.algorithm == "rime"
        assert robust.scoring.aggregation_strategy == "trimmed_mean"
        assert robust.semantic_analysis.model == "all-MiniLM-L6-v2"

    def test_research_profile_tier_3(self):
        """Test research profile has Tier 3 settings."""
        research = PROFILES["research"]
        assert research.tier == 3
        assert research.profile_name == "research"
        assert research.phonetic_analysis.algorithm == "hungarian"
        assert research.scoring.aggregation_strategy == "winsorized_mean"
        assert "mpnet" in research.semantic_analysis.model

    def test_scoring_weights_sum_to_one(self):
        """Test that scoring weights approximately sum to 1.0 in all profiles."""
        for profile_name, profile in PROFILES.items():
            total = (
                profile.scoring.phonetic_weight
                + profile.scoring.structural_weight
                + profile.scoring.semantic_weight
            )
            assert abs(total - 1.0) < 0.01, f"{profile_name} weights don't sum to 1.0"


class TestLoadConfig:
    """Tests for load_config() function."""

    def test_load_simple_profile(self):
        """Test loading simple profile without overrides."""
        config = load_config("simple")
        assert config.profile_name == "simple"
        assert config.tier == 1

    def test_load_robust_profile(self):
        """Test loading robust profile without overrides."""
        config = load_config("robust")
        assert config.profile_name == "robust"
        assert config.tier == 2

    def test_load_research_profile(self):
        """Test loading research profile without overrides."""
        config = load_config("research")
        assert config.profile_name == "research"
        assert config.tier == 3

    def test_load_invalid_profile_raises_error(self):
        """Test that loading non-existent profile raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            load_config("nonexistent")

    def test_default_profile_is_simple(self):
        """Test that load_config() defaults to simple profile."""
        config = load_config()
        assert config.profile_name == "simple"


class TestConfigOverrides:
    """Tests for configuration override functionality."""

    def test_override_single_field(self):
        """Test overriding a single config field."""
        config = load_config("simple", {"scoring.phonetic_weight": 0.5})
        assert config.scoring.phonetic_weight == 0.5
        # Verify other fields unchanged
        assert config.scoring.structural_weight == 0.33

    def test_override_multiple_fields(self):
        """Test overriding multiple config fields."""
        overrides = {
            "scoring.phonetic_weight": 0.5,
            "zone_extraction.window_size": 5,
            "phonetic_analysis.algorithm": "rime",
        }
        config = load_config("simple", overrides)
        assert config.scoring.phonetic_weight == 0.5
        assert config.zone_extraction.window_size == 5
        assert config.phonetic_analysis.algorithm == "rime"

    def test_override_top_level_field(self):
        """Test overriding top-level config field."""
        config = load_config("simple", {"tier": 2})
        assert config.tier == 2

    def test_override_invalid_field_raises_error(self):
        """Test that overriding non-existent field raises KeyError."""
        with pytest.raises(KeyError):
            load_config("simple", {"scoring.nonexistent_field": 0.5})

    def test_override_invalid_component_raises_error(self):
        """Test that overriding non-existent component raises KeyError."""
        with pytest.raises(KeyError):
            load_config("simple", {"nonexistent.field": 0.5})

    def test_override_preserves_original_profile(self):
        """Test that overrides don't modify the original PROFILES."""
        original_weight = PROFILES["simple"].scoring.phonetic_weight
        config = load_config("simple", {"scoring.phonetic_weight": 0.9})

        assert config.scoring.phonetic_weight == 0.9
        assert PROFILES["simple"].scoring.phonetic_weight == original_weight

    def test_override_with_different_types(self):
        """Test overriding fields with different value types."""
        overrides = {
            "scoring.phonetic_weight": 0.5,  # float
            "zone_extraction.window_size": 5,  # int
            "phonetic_analysis.algorithm": "rime",  # string
            "scoring.use_pair_confidence": True,  # bool
        }
        config = load_config("robust", overrides)
        assert isinstance(config.scoring.phonetic_weight, float)
        assert isinstance(config.zone_extraction.window_size, int)
        assert isinstance(config.phonetic_analysis.algorithm, str)
        assert isinstance(config.scoring.use_pair_confidence, bool)


class TestComponentConfigs:
    """Tests for individual component configuration classes."""

    def test_clause_detection_config_defaults(self):
        """Test ClauseDetectionConfig default values."""
        config = ClauseDetectionConfig()
        assert config.min_length == 3
        assert config.max_length == 50
        assert ";" in config.punctuation
        assert config.strict_mode is False

    def test_pair_rules_config_defaults(self):
        """Test PairRulesConfig default values."""
        config = PairRulesConfig()
        assert "but" in config.conjunctions
        assert "However," in config.transitions
        assert config.min_pair_confidence == 0.0
        assert config.use_confidence_weighting is False

    def test_scoring_config_weights(self):
        """Test ScoringConfig weight configuration."""
        config = ScoringConfig()
        assert 0.0 <= config.phonetic_weight <= 1.0
        assert 0.0 <= config.structural_weight <= 1.0
        assert 0.0 <= config.semantic_weight <= 1.0


class TestTierProgression:
    """Tests for tier progression features."""

    def test_tier_1_simple_algorithms(self):
        """Test Tier 1 uses simplest algorithms."""
        config = PROFILES["simple"]
        assert config.phonetic_analysis.algorithm == "levenshtein"
        assert config.scoring.aggregation_strategy == "mean"
        assert config.scoring.missing_data_strategy == "zero"
        assert config.clause_detection.cross_sentence_pairing is False

    def test_tier_2_enhanced_features(self):
        """Test Tier 2 enables enhanced features."""
        config = PROFILES["robust"]
        assert config.phonetic_analysis.top_k_matches > 1
        assert config.scoring.outlier_removal is True
        assert config.pair_rules.use_confidence_weighting is True
        assert config.zone_extraction.exclude_discourse_markers is True

    def test_tier_3_advanced_features(self):
        """Test Tier 3 enables advanced/experimental features."""
        config = PROFILES["research"]
        assert config.phonetic_analysis.use_stress_patterns is True
        assert config.phonetic_analysis.cache_results is True
        assert config.zone_extraction.adaptive_window is True
        assert config.clause_detection.cross_sentence_pairing is True
        assert config.validation.distribution_fitting is True

    def test_tier_progression_maintains_backward_compatibility(self):
        """Test that higher tiers don't break lower tier functionality."""
        simple = PROFILES["simple"]
        robust = PROFILES["robust"]
        research = PROFILES["research"]

        # All should have same basic structure
        for profile in [simple, robust, research]:
            assert hasattr(profile, "scoring")
            assert hasattr(profile, "phonetic_analysis")
            assert hasattr(profile, "semantic_analysis")
            assert hasattr(profile.scoring, "phonetic_weight")


class TestConfigDataFlow:
    """Integration tests for config usage in pipeline."""

    def test_config_access_pattern(self):
        """Test typical config access pattern used by components."""
        config = load_config("simple")

        # Access nested config values
        algorithm = config.phonetic_analysis.algorithm
        window_size = config.zone_extraction.window_size
        weights = {
            "phonetic": config.scoring.phonetic_weight,
            "structural": config.scoring.structural_weight,
            "semantic": config.scoring.semantic_weight,
        }

        assert isinstance(algorithm, str)
        assert isinstance(window_size, int)
        assert len(weights) == 3

    def test_config_for_component_initialization(self):
        """Test using config to initialize component settings."""
        config = load_config("robust")

        # Simulate component reading its config
        phonetic_settings = {
            "algorithm": config.phonetic_analysis.algorithm,
            "top_k": config.phonetic_analysis.top_k_matches,
            "penalty": config.phonetic_analysis.length_penalty,
        }

        assert phonetic_settings["algorithm"] == "rime"
        assert phonetic_settings["top_k"] == 2
        assert phonetic_settings["penalty"] == 0.1
