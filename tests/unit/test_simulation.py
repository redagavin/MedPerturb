# ABOUTME: Tests for power simulation across all 5 metrics
# ABOUTME: Validates generative model, detection rates, and null calibration

import numpy as np
import pandas as pd


class TestRunSingleSimulation:
    """Tests for run_single_simulation producing results for all 5 metrics."""

    def test_returns_all_five_metrics(self):
        """Single simulation returns a p-value dict for all 5 metrics."""
        from simulation import run_single_simulation, generate_responses
        rng = np.random.default_rng(42)
        z_i = rng.normal(0, 2, size=100)
        y_orig = (z_i > 0).astype(int)
        data = generate_responses(z_i, y_orig, 0.5, 0.3, rng)
        boot_rng = np.random.default_rng(99)
        result = run_single_simulation(data, n_bootstrap=100, rng=boot_rng)
        assert set(result.keys()) == {"mi", "phi", "flip_rate", "jsd", "kl"}

    def test_all_pvalues_between_0_and_1(self):
        """All p-values are valid."""
        from simulation import run_single_simulation, generate_responses
        rng = np.random.default_rng(42)
        z_i = rng.normal(0, 2, size=100)
        y_orig = (z_i > 0).astype(int)
        data = generate_responses(z_i, y_orig, 0.5, 0.3, rng)
        boot_rng = np.random.default_rng(99)
        result = run_single_simulation(data, n_bootstrap=100, rng=boot_rng)
        for metric, pval in result.items():
            assert 0.0 <= pval <= 1.0, f"{metric} p-value {pval} out of range"


class TestGenerateResponses:
    """Tests for generate_responses using real logits."""

    def _make_z_and_y(self, n=100, seed=42):
        """Create realistic z_i and y_orig arrays."""
        rng = np.random.default_rng(seed)
        z_i = rng.normal(0, 2, size=n)
        y_orig = (z_i > 0).astype(int)
        return z_i, y_orig

    def test_returns_required_keys(self):
        from simulation import generate_responses
        z_i, y_orig = self._make_z_and_y()
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        assert set(result.keys()) == {"orig", "pert", "base", "p_orig", "p_pert", "p_base"}

    def test_lengths_match_input(self):
        from simulation import generate_responses
        z_i, y_orig = self._make_z_and_y(n=50)
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        for key in result:
            assert len(result[key]) == 50

    def test_p_orig_is_sigmoid_of_z(self):
        from simulation import generate_responses
        from scipy.special import expit as sigmoid
        z_i = np.array([0.0, 1.0, -1.0, 3.0])
        y_orig = np.array([0, 1, 0, 1])
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        np.testing.assert_array_almost_equal(result["p_orig"], sigmoid(z_i))

    def test_orig_is_y_orig_not_resampled(self):
        from simulation import generate_responses
        z_i = np.array([0.0, 1.0, -1.0, 3.0])
        y_orig = np.array([0, 1, 0, 1])
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        np.testing.assert_array_equal(result["orig"], y_orig)

    def test_prob_vectors_in_0_1(self):
        from simulation import generate_responses
        z_i, y_orig = self._make_z_and_y()
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=1.0, sigma=0.5, rng=rng)
        for key in ["p_orig", "p_pert", "p_base"]:
            assert np.all(result[key] > 0.0)
            assert np.all(result[key] < 1.0)

    def test_binary_vectors_are_binary(self):
        from simulation import generate_responses
        z_i, y_orig = self._make_z_and_y()
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        for key in ["pert", "base"]:
            assert set(np.unique(result[key])).issubset({0, 1})

    def test_null_hypothesis_symmetric(self):
        from simulation import generate_responses
        z_i, y_orig = self._make_z_and_y(n=10000)
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=0.0, sigma=0.5, rng=rng)
        pert_diff = np.mean(np.abs(result["p_pert"] - result["p_orig"]))
        base_diff = np.mean(np.abs(result["p_base"] - result["p_orig"]))
        assert abs(pert_diff - base_diff) < 0.03

    def test_sigma_pert_adds_extra_noise_to_pert_arm(self):
        from simulation import generate_responses
        z_i, y_orig = self._make_z_and_y(n=10000)
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=1.0, sigma=0.3, rng=rng)
        pert_diff = np.mean(np.abs(result["p_pert"] - result["p_orig"]))
        base_diff = np.mean(np.abs(result["p_base"] - result["p_orig"]))
        assert pert_diff > base_diff + 0.01

    def test_zero_noise_preserves_original(self):
        from simulation import generate_responses
        z_i = np.array([0.0, 1.0, -1.0, 3.0])
        y_orig = np.array([0, 1, 0, 1])
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=0.0, sigma=0.0, rng=rng)
        np.testing.assert_array_almost_equal(result["p_pert"], result["p_orig"])
        np.testing.assert_array_almost_equal(result["p_base"], result["p_orig"])

    def test_deterministic_with_same_seed(self):
        from simulation import generate_responses
        z_i, y_orig = self._make_z_and_y()
        r1 = generate_responses(z_i, y_orig, 0.5, 0.3, np.random.default_rng(42))
        r2 = generate_responses(z_i, y_orig, 0.5, 0.3, np.random.default_rng(42))
        for key in r1:
            np.testing.assert_array_equal(r1[key], r2[key])

    def test_extreme_z_values(self):
        from simulation import generate_responses
        z_i = np.array([13.8, -13.8, 0.0, 13.8, -13.8])
        y_orig = np.array([1, 0, 0, 1, 0])
        rng = np.random.default_rng(42)
        result = generate_responses(z_i, y_orig, sigma_pert=1.0, sigma=0.5, rng=rng)
        for key in ["p_orig", "p_pert", "p_base"]:
            assert np.all(np.isfinite(result[key]))
            assert np.all(result[key] >= 0.0)
            assert np.all(result[key] <= 1.0)


class TestComboSeed:
    """Tests for deterministic per-combo seeding with formatted floats."""

    def test_same_params_same_seed(self):
        from simulation import _combo_seed
        s1 = _combo_seed(42, 0.5, 0.25, "MANAGE_8b")
        s2 = _combo_seed(42, 0.5, 0.25, "MANAGE_8b")
        assert s1 == s2

    def test_different_params_different_seed(self):
        from simulation import _combo_seed
        s1 = _combo_seed(42, 0.5, 0.25, "MANAGE_8b")
        s2 = _combo_seed(42, 0.6, 0.25, "MANAGE_8b")
        assert s1 != s2

    def test_different_conditions_different_seed(self):
        from simulation import _combo_seed
        s1 = _combo_seed(42, 0.5, 0.25, "MANAGE_8b")
        s2 = _combo_seed(42, 0.5, 0.25, "VISIT_8b")
        assert s1 != s2

    def test_float_formatting_canonical(self):
        from simulation import _combo_seed
        s1 = _combo_seed(42, 0.3, 0.25, "MANAGE_8b")
        s2 = _combo_seed(42, 0.30000000000000004, 0.25, "MANAGE_8b")
        assert s1 == s2

    def test_seed_in_valid_range(self):
        from simulation import _combo_seed
        s = _combo_seed(42, 0.5, 0.25, "MANAGE_8b")
        assert isinstance(s, int)
        assert 0 <= s < 2**31


class TestRunOneCombo:
    """Tests for _run_one_combo producing correct results structure."""

    def test_returns_list_of_5_metric_dicts(self):
        from simulation import _run_one_combo
        rng = np.random.default_rng(42)
        z_i = rng.normal(0, 1.5, size=50)
        y_orig = (z_i > 0).astype(int)
        args = (0.5, 0.3, "MANAGE_8b", 5, 50, 42, z_i, y_orig)
        results = _run_one_combo(args)
        assert len(results) == 5
        metrics = {r["metric"] for r in results}
        assert metrics == {"mi", "phi", "flip_rate", "jsd", "kl"}

    def test_result_dict_has_required_keys(self):
        from simulation import _run_one_combo
        rng = np.random.default_rng(42)
        z_i = rng.normal(0, 1.5, size=50)
        y_orig = (z_i > 0).astype(int)
        args = (0.5, 0.3, "MANAGE_8b", 5, 50, 42, z_i, y_orig)
        results = _run_one_combo(args)
        for r in results:
            assert r["sigma_pert"] == 0.5
            assert r["sigma"] == 0.3
            assert r["condition"] == "MANAGE_8b"
            assert 0.0 <= r["detection_rate"] <= 1.0
            assert 0.0 <= r["mean_p_value"] <= 1.0

    def test_deterministic(self):
        from simulation import _run_one_combo
        rng = np.random.default_rng(42)
        z_i = rng.normal(0, 1.5, size=50)
        y_orig = (z_i > 0).astype(int)
        args = (0.5, 0.3, "MANAGE_8b", 10, 50, 42, z_i, y_orig)
        r1 = _run_one_combo(args)
        r2 = _run_one_combo(args)
        for a, b in zip(r1, r2):
            assert a["detection_rate"] == b["detection_rate"]
            assert a["mean_p_value"] == b["mean_p_value"]


class TestRunPowerAnalysis:
    """Tests for the power analysis loop with real logits."""

    def _make_z_and_y(self, n=50, seed=42):
        rng = np.random.default_rng(seed)
        z_i = rng.normal(0, 1.5, size=n)
        y_orig = (z_i > 0).astype(int)
        return z_i, y_orig

    def test_output_columns(self):
        from simulation import run_power_analysis
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 0.5], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
        )
        required = {"metric", "sigma_pert", "sigma", "condition", "detection_rate", "mean_p_value"}
        assert required == set(results.columns)

    def test_output_row_count(self):
        from simulation import run_power_analysis
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 0.5, 1.0], sigma_values=[0.25, 0.5],
            n_simulations=5, n_bootstrap=50, seed=42,
        )
        assert len(results) == 30  # 5 metrics x 3 sigma_pert x 2 sigma

    def test_all_five_metrics(self):
        from simulation import run_power_analysis
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
        )
        assert set(results["metric"].unique()) == {"mi", "phi", "flip_rate", "jsd", "kl"}

    def test_detection_rate_between_0_and_1(self):
        from simulation import run_power_analysis
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 1.0], sigma_values=[0.3],
            n_simulations=10, n_bootstrap=50, seed=42,
        )
        assert (results["detection_rate"] >= 0).all()
        assert (results["detection_rate"] <= 1).all()

    def test_condition_column_populated(self):
        from simulation import run_power_analysis
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis(
            z_i=z_i, y_orig=y_orig, condition="VISIT_70b",
            sigma_pert_values=[0.0], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
        )
        assert (results["condition"] == "VISIT_70b").all()

    def test_null_calibration(self):
        from simulation import run_power_analysis
        z_i, y_orig = self._make_z_and_y(n=100)
        results = run_power_analysis(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0], sigma_values=[0.5],
            n_simulations=200, n_bootstrap=200, seed=42,
        )
        for metric in ["mi", "phi", "flip_rate", "jsd", "kl"]:
            rate = results[results["metric"] == metric]["detection_rate"].values[0]
            assert rate < 0.15, f"{metric} null rate {rate} too high (expected ~0.05)"

    def test_checkpoint_and_resume(self, tmp_path):
        from simulation import run_power_analysis
        z_i, y_orig = self._make_z_and_y()
        checkpoint = str(tmp_path / "ckpt.csv")

        full = run_power_analysis(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 0.5, 1.0], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
        )

        partial_rows = full[
            (full["sigma_pert"] == 0.0) & (full["sigma"] == 0.3)
        ].copy()
        partial_rows.to_csv(checkpoint, index=False)

        resumed = run_power_analysis(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 0.5, 1.0], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
            checkpoint_path=checkpoint,
        )

        assert len(resumed) == len(full)
        for _, frow in full.iterrows():
            match = resumed[
                (resumed["metric"] == frow["metric"]) &
                (resumed["sigma_pert"] == frow["sigma_pert"]) &
                (resumed["sigma"] == frow["sigma"])
            ]
            assert len(match) == 1
            assert match["detection_rate"].values[0] == frow["detection_rate"], (
                f"Mismatch for {frow['metric']} sp={frow['sigma_pert']} s={frow['sigma']}"
            )


class TestGeneratePowerCurves:
    """Tests for power curve figure generation."""

    def test_creates_png_file(self, tmp_path):
        from simulation import generate_power_curves
        rows = []
        for metric in ["mi", "phi", "flip_rate", "jsd", "kl"]:
            for sp in [0.0, 0.5, 1.0]:
                for sig in [0.25, 0.5]:
                    rows.append({
                        "metric": metric, "sigma_pert": sp, "sigma": sig,
                        "condition": "MANAGE_8b", "detection_rate": 0.05,
                        "mean_p_value": 0.5,
                    })
        results = pd.DataFrame(rows)
        generate_power_curves(results, str(tmp_path), "MANAGE_8b")
        assert (tmp_path / "power_curves_MANAGE_8b.png").exists()

    def test_no_error_on_single_sigma(self, tmp_path):
        from simulation import generate_power_curves
        rows = []
        for metric in ["mi", "phi", "flip_rate", "jsd", "kl"]:
            for sp in [0.0, 0.5]:
                rows.append({
                    "metric": metric, "sigma_pert": sp, "sigma": 0.5,
                    "condition": "MANAGE_8b", "detection_rate": 0.05,
                    "mean_p_value": 0.5,
                })
        results = pd.DataFrame(rows)
        generate_power_curves(results, str(tmp_path), "MANAGE_8b")
        assert (tmp_path / "power_curves_MANAGE_8b.png").exists()
