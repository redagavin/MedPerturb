# ABOUTME: Tests for bootstrap MI analysis
# ABOUTME: Verifies correct MI calculation and bootstrap hypothesis testing

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/case_studies')


class TestMutualInformation:
    """Tests for MI calculation (matching original notebook)."""

    def test_mi_identical_arrays(self):
        """MI of identical arrays should be maximum (entropy of the array)."""
        from baseline_analysis import calculate_mi

        x = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
        mi = calculate_mi(x, x)

        # MI(X,X) = H(X) = entropy of binary with p=0.5 = 1 bit
        assert abs(mi - 1.0) < 0.01

    def test_mi_independent_arrays(self):
        """MI of independent arrays should be near zero."""
        from baseline_analysis import calculate_mi

        np.random.seed(42)
        x = pd.Series(np.random.randint(0, 2, 1000))
        y = pd.Series(np.random.randint(0, 2, 1000))

        mi = calculate_mi(x, y)
        assert mi < 0.05  # Should be close to 0

    def test_mi_non_negative(self):
        """MI must always be non-negative."""
        from baseline_analysis import calculate_mi

        for _ in range(10):
            x = pd.Series(np.random.randint(0, 2, 100))
            y = pd.Series(np.random.randint(0, 2, 100))
            mi = calculate_mi(x, y)
            assert mi >= 0


class TestBootstrapMITest:
    """Tests for bootstrap hypothesis testing."""

    def test_significant_difference_detected(self):
        """Should detect significant difference when perturbation has larger effect."""
        from baseline_analysis import bootstrap_mi_test

        np.random.seed(42)
        n = 200

        # Original responses
        orig = pd.Series(np.random.randint(0, 2, n))

        # Perturbation causes many flips (high MI with original = low change)
        # Wait, MI measures dependency. If pert = orig, MI is high.
        # If pert is random, MI is low.
        # We want: perturbation causes MORE change than baseline
        # More change = LESS correlation = LOWER MI

        # Perturbation: 50% random (less correlated)
        pert = orig.copy()
        flip_mask = np.random.random(n) < 0.5
        pert[flip_mask] = 1 - pert[flip_mask]

        # Baseline: 20% random (more correlated)
        base = orig.copy()
        flip_mask = np.random.random(n) < 0.2
        base[flip_mask] = 1 - base[flip_mask]

        result = bootstrap_mi_test(orig, pert, base, n_bootstrap=500)

        # MI(orig, pert) should be LOWER than MI(orig, base)
        # because perturbation causes more change
        assert result['mi_perturbation'] < result['mi_baseline']
        assert result['observed_diff'] < 0

    def test_returns_confidence_interval(self):
        """Must return CI bounds."""
        from baseline_analysis import bootstrap_mi_test

        orig = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        pert = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        base = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])

        result = bootstrap_mi_test(orig, pert, base, n_bootstrap=100)

        assert 'ci_low' in result
        assert 'ci_high' in result
        assert result['ci_low'] <= result['ci_high']

    def test_returns_p_value(self):
        """Must return p-value between 0 and 1."""
        from baseline_analysis import bootstrap_mi_test

        orig = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        pert = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        base = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])

        result = bootstrap_mi_test(orig, pert, base, n_bootstrap=100)

        assert 0 <= result['p_value'] <= 1


class TestConversationalFilter:
    """baseline_analysis must exclude conversational rows."""

    def test_run_analysis_excludes_conversational(self, tmp_path):
        """Conversational rows must be excluded to prevent alignment contamination.

        Conversational data is a format perturbation, not a gender/uncertainty/etc.
        perturbation. Including it mixes perturbation types in the MI analysis.
        """
        from baseline_analysis import run_analysis

        rows = []
        # 5 non-conversational samples for gender_swap (did=1,2,6)
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
