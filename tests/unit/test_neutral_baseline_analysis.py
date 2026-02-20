# ABOUTME: Tests for neutral baseline MI analysis
# ABOUTME: Verifies MI calculation, bootstrap testing, and neutral baseline comparison logic

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/case_studies')


class TestMutualInformation:
    """Tests for MI calculation."""

    def test_mi_identical_arrays(self):
        """MI(X,X) = H(X) = 1.0 bit for balanced binary data."""
        from neutral_baseline_analysis import calculate_mi

        x = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
        mi = calculate_mi(x, x)

        assert abs(mi - 1.0) < 0.01

    def test_mi_independent_arrays(self):
        """MI of independent arrays should be near zero."""
        from neutral_baseline_analysis import calculate_mi

        np.random.seed(42)
        x = pd.Series(np.random.randint(0, 2, 1000))
        y = pd.Series(np.random.randint(0, 2, 1000))

        mi = calculate_mi(x, y)
        assert mi < 0.05

    def test_mi_non_negative(self):
        """MI must always be non-negative."""
        from neutral_baseline_analysis import calculate_mi

        for seed in range(10):
            np.random.seed(seed)
            x = pd.Series(np.random.randint(0, 2, 100))
            y = pd.Series(np.random.randint(0, 2, 100))
            mi = calculate_mi(x, y)
            assert mi >= 0


class TestBootstrapMITest:
    """Tests for bootstrap hypothesis testing."""

    def test_detects_significant_difference(self):
        """Should detect when perturbation causes more change than baseline."""
        from neutral_baseline_analysis import bootstrap_mi_test

        np.random.seed(42)
        n = 200

        orig = pd.Series(np.random.randint(0, 2, n))

        # Perturbation: 50% flip (less correlated, lower MI)
        pert = orig.copy()
        flip_mask = np.random.random(n) < 0.5
        pert[flip_mask] = 1 - pert[flip_mask]

        # Baseline: 20% flip (more correlated, higher MI)
        base = orig.copy()
        flip_mask = np.random.random(n) < 0.2
        base[flip_mask] = 1 - base[flip_mask]

        result = bootstrap_mi_test(orig, pert, base, n_bootstrap=500)

        # MI(orig, pert) < MI(orig, base) because perturbation causes more change
        assert result['observed_diff'] < 0

    def test_returns_confidence_interval(self):
        """CI bounds must be ordered: ci_low <= ci_high."""
        from neutral_baseline_analysis import bootstrap_mi_test

        orig = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        pert = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        base = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])

        result = bootstrap_mi_test(orig, pert, base, n_bootstrap=100)

        assert 'ci_low' in result
        assert 'ci_high' in result
        assert result['ci_low'] <= result['ci_high']

    def test_returns_valid_p_value(self):
        """p-value must be between 0 and 1."""
        from neutral_baseline_analysis import bootstrap_mi_test

        orig = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        pert = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        base = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])

        result = bootstrap_mi_test(orig, pert, base, n_bootstrap=100)

        assert 0 <= result['p_value'] <= 1


class TestConversationalFilter:
    """Neutral baseline analysis must exclude conversational rows."""

    def test_excludes_conversational_rows(self, tmp_path, monkeypatch):
        """Conversational rows must be excluded to prevent alignment contamination.

        Conversational data is a format perturbation, not a content perturbation.
        Including it mixes perturbation types in the MI analysis.
        """
        import neutral_baseline_analysis
        from neutral_baseline_analysis import run_analysis, bootstrap_mi_test

        # Reduce bootstrap iterations for test speed
        _orig_bootstrap = bootstrap_mi_test
        def _fast_bootstrap(orig, pert, base, n_bootstrap=50):
            return _orig_bootstrap(orig, pert, base, n_bootstrap=n_bootstrap)
        monkeypatch.setattr(neutral_baseline_analysis, 'bootstrap_mi_test', _fast_bootstrap)

        rows = []
        # 5 non-conversational samples with original (did=1), gender_swap (did=2), neutral (did=6)
        for i in range(5):
            for did in [1, 2, 6]:
                rows.append({
                    'dataset': 'askadoc', 'dataset_id': did, 'context_id': f'N{i}',
                    'LLAMA3_MANAGE': 1, 'LLAMA3_VISIT': 0, 'LLAMA3_RESOURCE': 1,
                    'LLAMA3-70_MANAGE': 1, 'LLAMA3-70_VISIT': 0, 'LLAMA3-70_RESOURCE': 1,
                })
        # 1 conversational sample (should be excluded)
        for did in [1, 2, 6]:
            rows.append({
                'dataset': 'conversational', 'dataset_id': did, 'context_id': '211',
                'LLAMA3_MANAGE': 0, 'LLAMA3_VISIT': 0, 'LLAMA3_RESOURCE': 0,
                'LLAMA3-70_MANAGE': 0, 'LLAMA3-70_VISIT': 0, 'LLAMA3-70_RESOURCE': 0,
            })

        csv_path = str(tmp_path / 'test.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        output_path = str(tmp_path / 'output.xlsx')
        run_analysis(csv_path, output_path)

        results = pd.read_excel(output_path)
        gs = results[results['perturbation_type'] == 'gender_swap']
        for _, row in gs.iterrows():
            assert row['n_cases'] == 5, (
                f"Expected 5 cases but got {int(row['n_cases'])}. "
                "Conversational rows may not be filtered."
            )


class TestRunAnalysis:
    """Tests for the full analysis pipeline."""

    def test_produces_results_for_all_perturbation_types(self, tmp_path, monkeypatch):
        """Output must contain results for all 4 perturbation types."""
        import neutral_baseline_analysis
        from neutral_baseline_analysis import run_analysis, bootstrap_mi_test

        # Reduce bootstrap iterations for test speed
        _orig_bootstrap = bootstrap_mi_test
        def _fast_bootstrap(orig, pert, base, n_bootstrap=50):
            return _orig_bootstrap(orig, pert, base, n_bootstrap=n_bootstrap)
        monkeypatch.setattr(neutral_baseline_analysis, 'bootstrap_mi_test', _fast_bootstrap)

        np.random.seed(123)
        rows = []
        # Create data for all perturbation types (2-5) plus neutral baseline (6)
        for i in range(10):
            for did in [1, 2, 3, 4, 5, 6]:
                rows.append({
                    'dataset': 'askadoc', 'dataset_id': did, 'context_id': f'C{i}',
                    'LLAMA3_MANAGE': np.random.randint(0, 2),
                    'LLAMA3_VISIT': np.random.randint(0, 2),
                    'LLAMA3_RESOURCE': np.random.randint(0, 2),
                    'LLAMA3-70_MANAGE': np.random.randint(0, 2),
                    'LLAMA3-70_VISIT': np.random.randint(0, 2),
                    'LLAMA3-70_RESOURCE': np.random.randint(0, 2),
                })

        csv_path = str(tmp_path / 'test.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        output_path = str(tmp_path / 'output.xlsx')
        run_analysis(csv_path, output_path)

        results = pd.read_excel(output_path)
        expected_types = {'gender_swap', 'gender_remove', 'uncertain', 'colorful'}
        actual_types = set(results['perturbation_type'].unique())

        assert actual_types == expected_types, (
            f"Expected perturbation types {expected_types} but got {actual_types}"
        )
