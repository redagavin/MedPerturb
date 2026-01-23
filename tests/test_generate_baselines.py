# ABOUTME: Tests for baseline generation functionality
# ABOUTME: Validates token change calculation for calibrated paraphrases

import pytest
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')

from generate_baselines import compute_token_change_percent


def test_compute_token_change_percent_identical():
    """Identical texts should have 0% change."""
    original = "A 45-year-old woman presents with chest pain."
    perturbed = "A 45-year-old woman presents with chest pain."
    # Using word-level comparison for testing without loading tokenizer
    result = compute_token_change_percent(original, perturbed, tokenizer=None, use_words=True)
    assert result == 0.0


def test_compute_token_change_percent_different():
    """Different texts should have non-zero change."""
    original = "A 45-year-old woman presents with chest pain."
    perturbed = "A 45-year-old man presents with chest pain."
    result = compute_token_change_percent(original, perturbed, tokenizer=None, use_words=True)
    assert result > 0.0
