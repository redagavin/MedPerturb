# ABOUTME: Tests for per-population metric modules (MI, Phi, Flip Rate)
# ABOUTME: Validates compute() and bootstrap_test() for each metric

import numpy as np
import pytest
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')


class TestMICompute:
    """MI between binary vectors."""

    def test_identical_vectors(self):
        """Identical vectors have maximum MI."""
        x = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        from metrics.mi import compute
        mi = compute(x, x)
        assert mi > 0.9  # near 1 bit for balanced binary

    def test_independent_vectors(self):
        """Uncorrelated vectors have MI near 0."""
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=10000)
        y = rng.integers(0, 2, size=10000)
        from metrics.mi import compute
        mi = compute(x, y)
        assert mi < 0.01

    def test_non_negative(self):
        """MI is always non-negative."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            x = rng.integers(0, 2, size=50)
            y = rng.integers(0, 2, size=50)
            from metrics.mi import compute
            assert compute(x, y) >= 0

    def test_constant_vector(self):
        """MI with a constant vector is 0."""
        x = np.array([1, 0, 1, 0, 1])
        y = np.ones(5, dtype=int)
        from metrics.mi import compute
        assert compute(x, y) == 0.0


class TestPhiCompute:
    """Phi correlation between binary vectors."""

    def test_identical_vectors(self):
        """Identical vectors have phi = 1."""
        x = np.array([0, 0, 1, 1, 0, 1])
        from metrics.phi import compute
        assert np.isclose(compute(x, x), 1.0)

    def test_opposite_vectors(self):
        """Opposite vectors have phi = -1."""
        x = np.array([0, 0, 1, 1, 0, 1])
        y = 1 - x
        from metrics.phi import compute
        assert np.isclose(compute(x, y), -1.0)

    def test_independent_vectors(self):
        """Uncorrelated vectors have phi near 0."""
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=10000)
        y = rng.integers(0, 2, size=10000)
        from metrics.phi import compute
        assert abs(compute(x, y)) < 0.05

    def test_constant_vector(self):
        """Phi with constant vector is 0 (undefined, convention)."""
        x = np.array([0, 1, 0, 1])
        y = np.ones(4, dtype=int)
        from metrics.phi import compute
        assert compute(x, y) == 0.0


class TestFlipRateCompute:
    """Flip rate between binary vectors."""

    def test_identical_vectors(self):
        """No flips when vectors are identical."""
        x = np.array([0, 1, 0, 1])
        from metrics.flip_rate import compute
        assert compute(x, x) == 0.0

    def test_all_flipped(self):
        """All flipped gives rate 1.0."""
        x = np.array([0, 0, 1, 1])
        y = 1 - x
        from metrics.flip_rate import compute
        assert compute(x, y) == 1.0

    def test_half_flipped(self):
        """Half flipped gives rate 0.5."""
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 1, 0])
        from metrics.flip_rate import compute
        assert compute(x, y) == 0.5


class TestBootstrapTest:
    """Bootstrap test shared across per-population metrics."""

    @pytest.mark.parametrize("metric_module", ["mi", "phi", "flip_rate"])
    def test_return_keys(self, metric_module):
        """All bootstrap tests return required keys."""
        rng = np.random.default_rng(42)
        orig = rng.integers(0, 2, size=50)
        pert = rng.integers(0, 2, size=50)
        base = rng.integers(0, 2, size=50)
        mod = __import__(f"metrics.{metric_module}", fromlist=["bootstrap_test"])
        result = mod.bootstrap_test(orig, pert, base, n_bootstrap=100, seed=42)
        expected_keys = {"statistic_perturbation", "statistic_baseline",
                         "observed_diff", "ci_low", "ci_high", "p_value"}
        assert set(result.keys()) == expected_keys

    @pytest.mark.parametrize("metric_module", ["mi", "phi", "flip_rate"])
    def test_p_value_range(self, metric_module):
        """p-value is in [0, 1]."""
        rng = np.random.default_rng(42)
        orig = rng.integers(0, 2, size=50)
        pert = rng.integers(0, 2, size=50)
        base = rng.integers(0, 2, size=50)
        mod = __import__(f"metrics.{metric_module}", fromlist=["bootstrap_test"])
        result = mod.bootstrap_test(orig, pert, base, n_bootstrap=100, seed=42)
        assert 0 <= result["p_value"] <= 1.0

    @pytest.mark.parametrize("metric_module", ["mi", "phi", "flip_rate"])
    def test_deterministic_with_seed(self, metric_module):
        """Same seed produces same result."""
        rng = np.random.default_rng(42)
        orig = rng.integers(0, 2, size=50)
        pert = rng.integers(0, 2, size=50)
        base = rng.integers(0, 2, size=50)
        mod = __import__(f"metrics.{metric_module}", fromlist=["bootstrap_test"])
        r1 = mod.bootstrap_test(orig, pert, base, n_bootstrap=200, seed=42)
        r2 = mod.bootstrap_test(orig, pert, base, n_bootstrap=200, seed=42)
        assert r1["p_value"] == r2["p_value"]

    @pytest.mark.parametrize("metric_module", ["mi", "phi", "flip_rate"])
    def test_ci_contains_observed(self, metric_module):
        """95% CI should usually contain the observed diff."""
        rng = np.random.default_rng(42)
        orig = rng.integers(0, 2, size=200)
        pert = rng.integers(0, 2, size=200)
        base = rng.integers(0, 2, size=200)
        mod = __import__(f"metrics.{metric_module}", fromlist=["bootstrap_test"])
        result = mod.bootstrap_test(orig, pert, base, n_bootstrap=500, seed=42)
        assert result["ci_low"] <= result["observed_diff"] <= result["ci_high"]

    @pytest.mark.parametrize("metric_module", ["mi", "phi", "flip_rate"])
    def test_empty_arrays_raise(self, metric_module):
        """Empty arrays should raise ValueError."""
        mod = __import__(f"metrics.{metric_module}", fromlist=["compute"])
        with pytest.raises((ValueError, ZeroDivisionError)):
            mod.compute(np.array([]), np.array([]))

    @pytest.mark.parametrize("metric_module", ["mi", "phi", "flip_rate"])
    def test_mismatched_lengths_raise(self, metric_module):
        """Mismatched array lengths should raise ValueError."""
        mod = __import__(f"metrics.{metric_module}", fromlist=["compute"])
        with pytest.raises(ValueError):
            mod.compute(np.array([0, 1, 0]), np.array([0, 1]))

    @pytest.mark.parametrize("metric_module", ["mi", "phi", "flip_rate"])
    def test_bootstrap_empty_arrays_raise(self, metric_module):
        """Bootstrap with empty arrays should raise ValueError."""
        mod = __import__(f"metrics.{metric_module}", fromlist=["bootstrap_test"])
        with pytest.raises((ValueError, ZeroDivisionError)):
            mod.bootstrap_test(np.array([]), np.array([]), np.array([]),
                               n_bootstrap=10, seed=42)

    @pytest.mark.parametrize("metric_module", ["mi", "phi", "flip_rate"])
    def test_non_binary_values_raise(self, metric_module):
        """Non-binary values (not 0 or 1) should raise ValueError."""
        mod = __import__(f"metrics.{metric_module}", fromlist=["compute"])
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            mod.compute(np.array([0, 1, 2, 3]), np.array([0, 1, 0, 1]))

    def test_mi_significant_when_pert_flips(self):
        """MI detects when perturbation causes flips but baseline doesn't."""
        rng = np.random.default_rng(42)
        orig = rng.integers(0, 2, size=200)
        pert = orig.copy()
        flip_mask = rng.random(200) < 0.7
        pert[flip_mask] = 1 - pert[flip_mask]
        base = orig.copy()
        flip_mask_base = rng.random(200) < 0.05
        base[flip_mask_base] = 1 - base[flip_mask_base]

        from metrics.mi import bootstrap_test
        result = bootstrap_test(orig, pert, base, n_bootstrap=500, seed=42)
        assert result["p_value"] < 0.05
