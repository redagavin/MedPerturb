# ABOUTME: Tests for power simulation across all 5 metrics
# ABOUTME: Validates generative model, detection rates, and null calibration

import numpy as np
import pandas as pd


class TestGenerateResponses:
    """Tests for generate_responses returning both binary and probability vectors."""

    def test_returns_binary_and_prob_vectors(self):
        """generate_responses returns a dict with binary and prob keys."""
        from simulation import generate_responses
        rng = np.random.default_rng(42)
        result = generate_responses(
            n_cases=100, beta0=0.0, beta1=2.0, beta_gender=0.5, rng=rng
        )
        assert "orig" in result
        assert "pert" in result
        assert "base" in result
        assert "p_orig" in result
        assert "p_pert" in result
        assert "p_base" in result

    def test_binary_vectors_are_binary(self):
        """Binary vectors contain only 0s and 1s."""
        from simulation import generate_responses
        rng = np.random.default_rng(42)
        result = generate_responses(
            n_cases=100, beta0=0.0, beta1=2.0, beta_gender=0.5, rng=rng
        )
        for key in ["orig", "pert", "base"]:
            assert set(np.unique(result[key])).issubset({0, 1})

    def test_prob_vectors_in_0_1(self):
        """Probability vectors are in [0, 1]."""
        from simulation import generate_responses
        rng = np.random.default_rng(42)
        result = generate_responses(
            n_cases=100, beta0=0.0, beta1=2.0, beta_gender=0.5, rng=rng
        )
        for key in ["p_orig", "p_pert", "p_base"]:
            assert np.all(result[key] >= 0.0)
            assert np.all(result[key] <= 1.0)

    def test_lengths_match(self):
        """All vectors have length n_cases."""
        from simulation import generate_responses
        rng = np.random.default_rng(42)
        result = generate_responses(
            n_cases=50, beta0=0.0, beta1=2.0, beta_gender=0.5, rng=rng
        )
        for key in ["orig", "pert", "base", "p_orig", "p_pert", "p_base"]:
            assert len(result[key]) == 50

    def test_deterministic_with_same_seed(self):
        """Same RNG produces identical output."""
        from simulation import generate_responses
        r1 = generate_responses(100, 0.0, 2.0, 0.5, np.random.default_rng(42))
        r2 = generate_responses(100, 0.0, 2.0, 0.5, np.random.default_rng(42))
        for key in ["orig", "pert", "base", "p_orig", "p_pert", "p_base"]:
            np.testing.assert_array_equal(r1[key], r2[key])

    def test_beta_gender_shifts_perturbation_prob(self):
        """Positive beta_gender increases perturbation probabilities."""
        from simulation import generate_responses
        r_null = generate_responses(10000, 0.0, 2.0, 0.0, np.random.default_rng(42))
        r_bias = generate_responses(10000, 0.0, 2.0, 2.0, np.random.default_rng(42))
        assert np.mean(r_bias["p_pert"]) > np.mean(r_null["p_pert"]) + 0.05

    def test_null_hypothesis_probs_similar(self):
        """When beta_gender=0, p_pert and p_base have same distribution."""
        from simulation import generate_responses
        result = generate_responses(
            10000, 0.0, 2.0, 0.0, np.random.default_rng(42)
        )
        # Under null, p_pert and p_orig should have same mean (no shift)
        assert abs(np.mean(result["p_pert"]) - np.mean(result["p_orig"])) < 0.01

    def test_both_arms_have_noise_when_sigma_positive(self):
        """Both perturbation and baseline arms get independent noise at sigma>0."""
        from simulation import generate_responses
        result = generate_responses(
            1000, 0.0, 2.0, 0.0, np.random.default_rng(42), sigma=0.3
        )
        # Under null with sigma>0, both arms should differ from orig
        pert_diff = np.mean(np.abs(result["p_pert"] - result["p_orig"]))
        base_diff = np.mean(np.abs(result["p_base"] - result["p_orig"]))
        assert pert_diff > 0.01, "p_pert should differ from p_orig when sigma>0"
        assert base_diff > 0.01, "p_base should differ from p_orig when sigma>0"

    def test_null_symmetric_noise_at_sigma_positive(self):
        """Under H0 with sigma>0, pert and base arms have similar mean divergence from orig."""
        from simulation import generate_responses
        result = generate_responses(
            10000, 0.0, 2.0, 0.0, np.random.default_rng(42), sigma=0.3
        )
        # Both arms should have similar mean absolute difference from orig
        pert_diff = np.mean(np.abs(result["p_pert"] - result["p_orig"]))
        base_diff = np.mean(np.abs(result["p_base"] - result["p_orig"]))
        assert abs(pert_diff - base_diff) < 0.02, (
            f"Under H0, both arms should have similar noise: "
            f"pert_diff={pert_diff:.4f}, base_diff={base_diff:.4f}"
        )


class TestRunSingleSimulation:
    """Tests for run_single_simulation producing results for all 5 metrics."""

    def test_returns_all_five_metrics(self):
        """Single simulation returns a p-value dict for all 5 metrics."""
        from simulation import run_single_simulation, generate_responses
        rng = np.random.default_rng(42)
        data = generate_responses(100, 0.0, 2.0, 1.0, rng)
        boot_rng = np.random.default_rng(99)
        result = run_single_simulation(data, n_bootstrap=100, rng=boot_rng)
        assert set(result.keys()) == {"mi", "phi", "flip_rate", "jsd", "kl"}

    def test_all_pvalues_between_0_and_1(self):
        """All p-values are valid."""
        from simulation import run_single_simulation, generate_responses
        rng = np.random.default_rng(42)
        data = generate_responses(100, 0.0, 2.0, 1.0, rng)
        boot_rng = np.random.default_rng(99)
        result = run_single_simulation(data, n_bootstrap=100, rng=boot_rng)
        for metric, pval in result.items():
            assert 0.0 <= pval <= 1.0, f"{metric} p-value {pval} out of range"


class TestRunPowerAnalysis:
    """Tests for the full power analysis loop."""

    def test_all_five_metrics(self):
        """Power analysis produces results for all 5 metrics."""
        from simulation import run_power_analysis
        results = run_power_analysis(
            beta1_values=[2.0], beta_gender_values=[0.0, 1.0],
            n_simulations=10, n_cases=50, n_bootstrap=100, seed=42,
        )
        metrics = set(results["metric"].unique())
        assert metrics == {"mi", "phi", "flip_rate", "jsd", "kl"}

    def test_output_columns(self):
        """Output DataFrame has required columns."""
        from simulation import run_power_analysis
        results = run_power_analysis(
            beta1_values=[2.0], beta_gender_values=[0.0],
            n_simulations=10, n_cases=50, n_bootstrap=100, seed=42,
        )
        required = {"metric", "beta1", "beta_gender", "detection_rate"}
        assert required.issubset(set(results.columns))

    def test_output_row_count(self):
        """One row per (metric, beta1, sigma, beta_gender) combination."""
        from simulation import run_power_analysis
        results = run_power_analysis(
            beta1_values=[1.0, 2.0], beta_gender_values=[0.0, 0.5],
            n_simulations=10, n_cases=50, n_bootstrap=100, seed=42,
        )
        # 5 metrics x 2 beta1 x 1 sigma x 2 beta_gender = 20 rows
        assert len(results) == 20

    def test_detection_rate_between_0_and_1(self):
        """Detection rates are valid proportions."""
        from simulation import run_power_analysis
        results = run_power_analysis(
            beta1_values=[2.0], beta_gender_values=[0.0, 1.0],
            n_simulations=10, n_cases=50, n_bootstrap=100, seed=42,
        )
        assert (results["detection_rate"] >= 0).all()
        assert (results["detection_rate"] <= 1).all()

    def test_detection_rate_increases_with_effect(self):
        """Detection rate increases with beta_gender for all metrics."""
        from simulation import run_power_analysis
        results = run_power_analysis(
            beta1_values=[2.0], beta_gender_values=[0.0, 2.0, 4.0],
            n_simulations=50, n_cases=100, n_bootstrap=200, seed=42,
        )
        for metric in ["mi", "phi", "flip_rate", "jsd", "kl"]:
            metric_rows = results[results["metric"] == metric]
            rates = metric_rows.sort_values("beta_gender")["detection_rate"].values
            assert rates[-1] >= rates[0], (
                f"{metric}: rate at beta_gender=4 ({rates[-1]}) < "
                f"rate at beta_gender=0 ({rates[0]})"
            )

    def test_null_detection_rate_near_alpha(self):
        """Under null hypothesis (beta_gender=0), detection rate ~ alpha (0.05).

        Uses sigma=0.3 so JSD/KL exercise non-degenerate paired differences.
        """
        from simulation import run_power_analysis
        results = run_power_analysis(
            beta1_values=[2.0], beta_gender_values=[0.0],
            sigma_values=[0.3],
            n_simulations=200, n_cases=100, n_bootstrap=200, seed=42,
        )
        for metric in ["mi", "phi", "flip_rate", "jsd", "kl"]:
            metric_rows = results[results["metric"] == metric]
            null_rate = metric_rows["detection_rate"].values[0]
            assert null_rate < 0.20, (
                f"{metric} null detection rate {null_rate} too high (expected ~0.05)"
            )

    def test_sigma_values_expand_grid(self):
        """sigma_values adds a dimension to the parameter grid."""
        from simulation import run_power_analysis
        results = run_power_analysis(
            beta1_values=[2.0], beta_gender_values=[0.0, 0.5],
            sigma_values=[0.0, 0.3],
            n_simulations=10, n_cases=50, n_bootstrap=100, seed=42,
        )
        # 5 metrics x 1 beta1 x 2 sigma x 2 beta_gender = 20 rows
        assert len(results) == 20
        assert "sigma" in results.columns
