# ABOUTME: Tests for baseline analysis with all 5 metrics, BH correction, both baselines
# ABOUTME: Verifies benjamini_hochberg, run_analysis JSON loading, and metric integration

import pytest
import numpy as np
import json
import os
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/case_studies')


def _read_source():
    """Read baseline_analysis.py source for inspection tests."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'case_studies', 'baseline_analysis.py')
    with open(path) as f:
        return f.read()


class TestBenjaminiHochberg:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_single_pvalue_unchanged(self):
        """Single p-value should be returned unchanged."""
        from baseline_analysis import benjamini_hochberg
        result = benjamini_hochberg(np.array([0.03]))
        assert np.isclose(result[0], 0.03)

    def test_all_significant_remain_significant(self):
        """Very small p-values should remain significant after correction."""
        from baseline_analysis import benjamini_hochberg
        pvals = np.array([0.001, 0.002, 0.003])
        adjusted = benjamini_hochberg(pvals)
        assert all(p < 0.05 for p in adjusted)

    def test_adjusted_geq_raw(self):
        """Adjusted p-values must be >= raw p-values."""
        from baseline_analysis import benjamini_hochberg
        pvals = np.array([0.01, 0.04, 0.03, 0.08, 0.5])
        adjusted = benjamini_hochberg(pvals)
        assert all(a >= r - 1e-10 for a, r in zip(adjusted, pvals))

    def test_adjusted_capped_at_one(self):
        """Adjusted p-values must not exceed 1.0."""
        from baseline_analysis import benjamini_hochberg
        pvals = np.array([0.5, 0.7, 0.9])
        adjusted = benjamini_hochberg(pvals)
        assert all(p <= 1.0 for p in adjusted)

    def test_monotonicity_of_adjusted(self):
        """Sorted adjusted p-values must be non-decreasing."""
        from baseline_analysis import benjamini_hochberg
        pvals = np.array([0.01, 0.04, 0.03, 0.08, 0.5])
        adjusted = benjamini_hochberg(pvals)
        sorted_adj = np.sort(adjusted)
        assert all(sorted_adj[i] <= sorted_adj[i + 1] + 1e-10
                   for i in range(len(sorted_adj) - 1))

    def test_known_example(self):
        """Verify against hand-calculated BH correction values."""
        from baseline_analysis import benjamini_hochberg
        pvals = np.array([0.01, 0.02, 0.03, 0.10, 0.50])
        adjusted = benjamini_hochberg(pvals)
        assert np.isclose(adjusted[0], 0.05)
        assert np.isclose(adjusted[1], 0.05)
        assert np.isclose(adjusted[2], 0.05)
        assert np.isclose(adjusted[3], 0.125)
        assert np.isclose(adjusted[4], 0.50)


class TestUsesAllMetrics:
    """Verify analysis script imports all 5 metric modules."""

    def test_imports_all_metrics(self):
        """Source must reference all 5 metric modules."""
        source = _read_source()
        for metric in ['mi', 'phi', 'flip_rate', 'jsd', 'kl']:
            assert metric in source, f"Missing metric: {metric}"


class TestRunAnalysisJSON:
    """Tests for run_analysis loading JSON evaluation results."""

    def _make_eval_results(self, n_samples=10, seed=42):
        """Create synthetic main_evaluate.py-format JSON evaluation results.

        Each sample has:
        - context_id
        - original_{TASK}: {binary_answers, logit_probs}
        - {perturbation}_{TASK}: {binary_answers, logit_probs}
        - {perturbation}_baseline_{TASK}: {binary_answers, logit_probs}
        - neutral_{TASK}: {binary_answers, logit_probs}
        """
        rng = np.random.default_rng(seed)
        perturbations = ['gender_swap', 'gender_remove', 'uncertain', 'colorful']
        tasks = ['MANAGE', 'VISIT', 'RESOURCE']

        results = []
        for i in range(n_samples):
            sample = {'context_id': i}
            for task in tasks:
                # Original
                answers = rng.integers(0, 2, 3).tolist()
                sample[f'original_{task}'] = {
                    'binary_answers': answers,
                    'logit_probs': float(rng.uniform(0.1, 0.9)),
                }
                # Perturbations + calibrated baselines
                for pert in perturbations:
                    answers = rng.integers(0, 2, 3).tolist()
                    sample[f'{pert}_{task}'] = {
                        'binary_answers': answers,
                        'logit_probs': float(rng.uniform(0.1, 0.9)),
                    }
                    answers = rng.integers(0, 2, 3).tolist()
                    sample[f'{pert}_baseline_{task}'] = {
                        'binary_answers': answers,
                        'logit_probs': float(rng.uniform(0.1, 0.9)),
                    }
                # Neutral baseline
                answers = rng.integers(0, 2, 3).tolist()
                sample[f'neutral_{task}'] = {
                    'binary_answers': answers,
                    'logit_probs': float(rng.uniform(0.1, 0.9)),
                }
            results.append(sample)
        return results

    def test_run_analysis_loads_json(self, tmp_path):
        """run_analysis must accept JSON evaluation results and produce output."""
        from baseline_analysis import run_analysis

        eval_results = self._make_eval_results(n_samples=20)
        eval_path = str(tmp_path / 'eval_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f)

        output_path = str(tmp_path / 'output.xlsx')
        run_analysis(eval_path, output_path)

        assert os.path.exists(output_path)

    def test_output_has_all_metrics(self, tmp_path):
        """Output must contain results for all 5 metrics."""
        from baseline_analysis import run_analysis
        import pandas as pd

        eval_results = self._make_eval_results(n_samples=20)
        eval_path = str(tmp_path / 'eval_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f)

        output_path = str(tmp_path / 'output.xlsx')
        run_analysis(eval_path, output_path)

        results = pd.read_excel(output_path)
        metrics_in_output = set(results['metric'].unique())
        expected_metrics = {'mi', 'phi', 'flip_rate', 'jsd', 'kl'}
        assert expected_metrics == metrics_in_output, (
            f"Expected metrics {expected_metrics} but got {metrics_in_output}"
        )

    def test_output_has_both_baselines(self, tmp_path):
        """Output must contain results for both calibrated and neutral baselines."""
        from baseline_analysis import run_analysis
        import pandas as pd

        eval_results = self._make_eval_results(n_samples=20)
        eval_path = str(tmp_path / 'eval_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f)

        output_path = str(tmp_path / 'output.xlsx')
        run_analysis(eval_path, output_path)

        results = pd.read_excel(output_path)
        baselines_in_output = set(results['baseline_type'].unique())
        expected_baselines = {'calibrated', 'neutral'}
        assert expected_baselines == baselines_in_output, (
            f"Expected baselines {expected_baselines} but got {baselines_in_output}"
        )

    def test_output_has_all_perturbation_types(self, tmp_path):
        """Output must contain results for all 4 perturbation types."""
        from baseline_analysis import run_analysis
        import pandas as pd

        eval_results = self._make_eval_results(n_samples=20)
        eval_path = str(tmp_path / 'eval_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f)

        output_path = str(tmp_path / 'output.xlsx')
        run_analysis(eval_path, output_path)

        results = pd.read_excel(output_path)
        expected_types = {'gender_swap', 'gender_remove', 'uncertain', 'colorful'}
        actual_types = set(results['perturbation_type'].unique())
        assert expected_types == actual_types, (
            f"Expected perturbation types {expected_types} but got {actual_types}"
        )

    def test_output_has_bh_adjusted_pvalues(self, tmp_path):
        """Output must contain BH-adjusted p-values."""
        from baseline_analysis import run_analysis
        import pandas as pd

        eval_results = self._make_eval_results(n_samples=20)
        eval_path = str(tmp_path / 'eval_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f)

        output_path = str(tmp_path / 'output.xlsx')
        run_analysis(eval_path, output_path)

        results = pd.read_excel(output_path)
        assert 'p_value_bh' in results.columns, "Missing BH-adjusted p-value column"
        assert all(results['p_value_bh'] >= results['p_value'] - 1e-10), (
            "BH-adjusted p-values must be >= raw p-values"
        )

    def test_output_has_required_columns(self, tmp_path):
        """Output must contain all required analysis columns."""
        from baseline_analysis import run_analysis
        import pandas as pd

        eval_results = self._make_eval_results(n_samples=20)
        eval_path = str(tmp_path / 'eval_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f)

        output_path = str(tmp_path / 'output.xlsx')
        run_analysis(eval_path, output_path)

        results = pd.read_excel(output_path)
        required_cols = [
            'perturbation_type', 'baseline_type', 'task', 'metric',
            'observed_diff', 'p_value', 'p_value_bh', 'n_cases',
        ]
        for col in required_cols:
            assert col in results.columns, f"Missing column: {col}"
