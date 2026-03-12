# ABOUTME: Tests for precision check analysis (age swap negative control)
# ABOUTME: Validates all 5 metrics, both baselines, result structure

import pytest
import numpy as np
import os
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/case_studies')


def _read_source():
    """Read precision_check_analysis.py source for inspection tests."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'case_studies', 'precision_check_analysis.py')
    with open(path) as f:
        return f.read()


class TestMajorityVote:
    """Tests for majority voting across seeds."""

    def test_majority_yes(self):
        from precision_check_analysis import majority_vote
        assert majority_vote([1, 1, 0]) == 1

    def test_majority_no(self):
        from precision_check_analysis import majority_vote
        assert majority_vote([0, 0, 1]) == 0

    def test_unanimous_yes(self):
        from precision_check_analysis import majority_vote
        assert majority_vote([1, 1, 1]) == 1

    def test_unanimous_no(self):
        from precision_check_analysis import majority_vote
        assert majority_vote([0, 0, 0]) == 0


class TestUsesAllMetrics:
    """Verify analysis script imports all 5 metric modules."""

    def test_imports_all_metrics(self):
        """Source must reference all 5 metric modules."""
        source = _read_source()
        for metric in ['mi', 'phi', 'flip_rate', 'jsd', 'kl']:
            assert metric in source, f"Missing metric: {metric}"


class TestPrecisionCheckAnalysis:
    """Tests for the full analysis pipeline with all metrics."""

    def _make_eval_results(self, n=100, seed=42):
        """Create synthetic precision check evaluation results.

        For negative control: age swap should NOT affect gender detection,
        so age_swap answers are mostly the same as original.
        """
        rng = np.random.default_rng(seed)
        eval_results = []
        for i in range(n):
            is_male = 1 if i % 2 == 0 else 0
            # Age swap preserves gender (5% noise)
            age_swap_val = (1 - is_male) if rng.random() < 0.05 else is_male
            # Neutral adds some noise (10% flip)
            neutral_val = (1 - is_male) if rng.random() < 0.1 else is_male
            eval_results.append({
                'context_id': f'N{i}',
                'original_GENDER': {
                    'binary_answers': [is_male, is_male, is_male],
                    'logit_probs': 0.9 if is_male else 0.1,
                },
                'age_swap_GENDER': {
                    'binary_answers': [age_swap_val, age_swap_val, age_swap_val],
                    'logit_probs': 0.85 if age_swap_val else 0.15,
                },
                'age_swap_baseline_GENDER': {
                    'binary_answers': [is_male, is_male, is_male],
                    'logit_probs': 0.9 if is_male else 0.1,
                },
                'neutral_GENDER': {
                    'binary_answers': [neutral_val, neutral_val, neutral_val],
                    'logit_probs': 0.8 if neutral_val == is_male else 0.3,
                },
            })
        return eval_results

    def test_returns_results_for_all_metrics(self):
        """Result dict must have entries for all 5 metrics."""
        from precision_check_analysis import run_precision_check_analysis

        eval_results = self._make_eval_results()
        results = run_precision_check_analysis(eval_results)

        expected_metrics = {'mi', 'phi', 'flip_rate', 'jsd', 'kl'}
        actual_metrics = set(results.keys())
        assert expected_metrics.issubset(actual_metrics), (
            f"Missing metrics: {expected_metrics - actual_metrics}"
        )

    def test_returns_results_for_both_baselines(self):
        """Each metric must have results for both calibrated and neutral baselines."""
        from precision_check_analysis import run_precision_check_analysis

        eval_results = self._make_eval_results()
        results = run_precision_check_analysis(eval_results)

        for metric in ['mi', 'phi', 'flip_rate', 'jsd', 'kl']:
            metric_result = results[metric]
            assert 'calibrated' in metric_result, (
                f"Missing calibrated baseline for {metric}"
            )
            assert 'neutral' in metric_result, (
                f"Missing neutral baseline for {metric}"
            )

    def test_each_metric_has_p_value(self):
        """Each metric x baseline result must have p_value."""
        from precision_check_analysis import run_precision_check_analysis

        eval_results = self._make_eval_results()
        results = run_precision_check_analysis(eval_results)

        for metric in ['mi', 'phi', 'flip_rate', 'jsd', 'kl']:
            for baseline in ['calibrated', 'neutral']:
                result = results[metric][baseline]
                assert 'p_value' in result, (
                    f"Missing p_value for {metric}/{baseline}"
                )
                assert 0 <= result['p_value'] <= 1

    def test_each_metric_has_observed_diff(self):
        """Each metric x baseline result must have observed_diff."""
        from precision_check_analysis import run_precision_check_analysis

        eval_results = self._make_eval_results()
        results = run_precision_check_analysis(eval_results)

        for metric in ['mi', 'phi', 'flip_rate', 'jsd', 'kl']:
            for baseline in ['calibrated', 'neutral']:
                result = results[metric][baseline]
                assert 'observed_diff' in result, (
                    f"Missing observed_diff for {metric}/{baseline}"
                )

    def test_n_cases_populated(self):
        """Result must contain n_cases."""
        from precision_check_analysis import run_precision_check_analysis

        eval_results = self._make_eval_results(n=50)
        results = run_precision_check_analysis(eval_results)
        assert results['n_cases'] == 50
