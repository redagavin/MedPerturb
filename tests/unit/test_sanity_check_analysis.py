# ABOUTME: Tests for sanity check MI analysis
# ABOUTME: Verifies majority voting and MI computation

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/case_studies')


class TestMajorityVote:
    """Tests for majority voting across seeds."""

    def test_majority_yes(self):
        from sanity_check_analysis import majority_vote
        assert majority_vote([1, 1, 0]) == 1

    def test_majority_no(self):
        from sanity_check_analysis import majority_vote
        assert majority_vote([0, 0, 1]) == 0

    def test_unanimous_yes(self):
        from sanity_check_analysis import majority_vote
        assert majority_vote([1, 1, 1]) == 1

    def test_unanimous_no(self):
        from sanity_check_analysis import majority_vote
        assert majority_vote([0, 0, 0]) == 0


class TestSanityCheckAnalysis:
    """Tests for the full MI analysis pipeline."""

    def test_strong_sanity_check(self):
        """When swap mostly flips answers and baseline preserves them,
        MI(orig,swap) should be lower than MI(orig,baseline).

        Note: MI measures statistical dependence (not direction), so a
        perfect flip (anti-correlation) has the same MI as perfect
        correlation. In real data, the swap introduces noise because
        models don't always detect the gender change, making MI lower.
        """
        from sanity_check_analysis import run_sanity_check_analysis

        np.random.seed(123)
        eval_results = []
        for i in range(100):
            is_male = 1 if i % 2 == 0 else 0
            is_female = 1 - is_male
            # Swap flips 70% of the time (realistic model behavior)
            swap_val = is_female if np.random.random() < 0.7 else is_male
            eval_results.append({
                'context_id': f'N{i}',
                'original_GENDER': {'binary_answers': [is_male, is_male, is_male]},
                'gender_swap_GENDER': {'binary_answers': [swap_val, swap_val, swap_val]},
                'gender_swap_baseline_GENDER': {'binary_answers': [is_male, is_male, is_male]},
            })

        result = run_sanity_check_analysis(eval_results)

        assert result['mi_perturbation'] < result['mi_baseline']
        assert result['observed_diff'] < 0
        assert result['n_cases'] == 100

    def test_with_some_noise(self):
        """Even with noise, the effect should be detectable."""
        from sanity_check_analysis import run_sanity_check_analysis

        np.random.seed(42)
        eval_results = []
        for i in range(100):
            is_male = 1 if i % 2 == 0 else 0
            is_female = 1 - is_male
            # Swap flips ~80% of the time (noise in model responses)
            swap_val = is_female if np.random.random() < 0.8 else is_male
            eval_results.append({
                'context_id': f'N{i}',
                'original_GENDER': {'binary_answers': [is_male, is_male, is_male]},
                'gender_swap_GENDER': {'binary_answers': [swap_val, swap_val, swap_val]},
                'gender_swap_baseline_GENDER': {'binary_answers': [is_male, is_male, is_male]},
            })

        result = run_sanity_check_analysis(eval_results)
        assert result['mi_perturbation'] < result['mi_baseline']

    def test_returns_required_fields(self):
        """Result must contain all required analysis fields."""
        from sanity_check_analysis import run_sanity_check_analysis

        eval_results = [
            {'context_id': f'N{i}',
             'original_GENDER': {'binary_answers': [i % 2, i % 2, i % 2]},
             'gender_swap_GENDER': {'binary_answers': [1 - i % 2, 1 - i % 2, 1 - i % 2]},
             'gender_swap_baseline_GENDER': {'binary_answers': [i % 2, i % 2, i % 2]}}
            for i in range(20)
        ]

        result = run_sanity_check_analysis(eval_results)
        required_keys = ['mi_perturbation', 'mi_baseline', 'observed_diff',
                         'ci_low', 'ci_high', 'p_value', 'n_cases']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
