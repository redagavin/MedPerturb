# ABOUTME: Tests for per-sample metric modules (JSD, KL Divergence)
# ABOUTME: Validates compute() and paired_ttest() for each metric

import numpy as np
import pytest
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')


class TestJSDCompute:
    """JSD between binary distributions."""

    def test_identical_distributions(self):
        """JSD of identical distributions is 0."""
        from metrics.jsd import compute
        assert np.isclose(compute(0.7, 0.7), 0.0)

    def test_symmetric(self):
        """JSD is symmetric: JSD(P,Q) = JSD(Q,P)."""
        from metrics.jsd import compute
        assert np.isclose(compute(0.8, 0.3), compute(0.3, 0.8))

    def test_maximum_divergence(self):
        """JSD is maximum (1.0 bit) for [1,0] vs [0,1]."""
        from metrics.jsd import compute
        assert np.isclose(compute(1.0, 0.0), 1.0)
        assert np.isclose(compute(0.0, 1.0), 1.0)

    def test_range(self):
        """JSD is in [0, 1] for binary distributions (log base 2)."""
        from metrics.jsd import compute
        rng = np.random.default_rng(42)
        for _ in range(50):
            p1, p2 = rng.uniform(0.01, 0.99, size=2)
            jsd = compute(p1, p2)
            assert 0 <= jsd <= 1.0 + 1e-10


class TestKLCompute:
    """KL divergence with epsilon floor."""

    def test_identical_distributions(self):
        """KL of identical distributions is 0."""
        from metrics.kl import compute
        assert np.isclose(compute(0.7, 0.7), 0.0)

    def test_asymmetric(self):
        """KL is NOT symmetric: KL(P||Q) != KL(Q||P) in general."""
        from metrics.kl import compute
        kl_pq = compute(0.8, 0.3)
        kl_qp = compute(0.3, 0.8)
        assert not np.isclose(kl_pq, kl_qp)

    def test_non_negative(self):
        """KL divergence is always non-negative."""
        from metrics.kl import compute
        rng = np.random.default_rng(42)
        for _ in range(50):
            p1, p2 = rng.uniform(0.01, 0.99, size=2)
            assert compute(p1, p2) >= -1e-10

    def test_epsilon_prevents_infinity(self):
        """Epsilon floor prevents infinity when Q has zeros."""
        from metrics.kl import compute
        # P(Yes)=0.5, P(No)=0.5 vs P(Yes)=1.0, P(No)=0.0
        kl = compute(0.5, 1.0)
        assert np.isfinite(kl)


class TestPairedTtest:
    """Paired t-test shared across per-sample metrics."""

    @pytest.mark.parametrize("metric_module", ["jsd", "kl"])
    def test_return_keys(self, metric_module):
        """All paired t-tests return required keys."""
        rng = np.random.default_rng(42)
        orig = rng.uniform(0.2, 0.8, size=50)
        pert = rng.uniform(0.2, 0.8, size=50)
        base = rng.uniform(0.2, 0.8, size=50)
        mod = __import__(f"metrics.{metric_module}", fromlist=["paired_ttest"])
        result = mod.paired_ttest(orig, pert, base)
        expected_keys = {"mean_perturbation", "mean_baseline",
                         "observed_diff", "t_statistic", "p_value"}
        assert set(result.keys()) == expected_keys

    @pytest.mark.parametrize("metric_module", ["jsd", "kl"])
    def test_p_value_range(self, metric_module):
        """p-value is in [0, 1]."""
        rng = np.random.default_rng(42)
        orig = rng.uniform(0.2, 0.8, size=50)
        pert = rng.uniform(0.2, 0.8, size=50)
        base = rng.uniform(0.2, 0.8, size=50)
        mod = __import__(f"metrics.{metric_module}", fromlist=["paired_ttest"])
        result = mod.paired_ttest(orig, pert, base)
        assert 0 <= result["p_value"] <= 1.0

    @pytest.mark.parametrize("metric_module", ["jsd", "kl"])
    def test_empty_arrays_raise(self, metric_module):
        """Empty arrays should raise ValueError."""
        mod = __import__(f"metrics.{metric_module}", fromlist=["paired_ttest"])
        with pytest.raises(ValueError):
            mod.paired_ttest(np.array([]), np.array([]), np.array([]))

    @pytest.mark.parametrize("metric_module", ["jsd", "kl"])
    def test_out_of_bounds_probs_raise(self, metric_module):
        """Probabilities outside [0, 1] should raise ValueError."""
        mod = __import__(f"metrics.{metric_module}", fromlist=["paired_ttest"])
        valid = np.array([0.5, 0.6, 0.7])
        with pytest.raises(ValueError, match="Probabilities must be in"):
            mod.paired_ttest(np.array([0.5, 1.5, 0.3]), valid, valid)
        with pytest.raises(ValueError, match="Probabilities must be in"):
            mod.paired_ttest(valid, np.array([-0.1, 0.5, 0.5]), valid)
        with pytest.raises(ValueError, match="Probabilities must be in"):
            mod.paired_ttest(valid, valid, np.array([0.5, 0.5, 2.0]))

    @pytest.mark.parametrize("metric_module", ["jsd", "kl"])
    def test_mismatched_lengths_raise(self, metric_module):
        """Mismatched array lengths should raise ValueError."""
        mod = __import__(f"metrics.{metric_module}", fromlist=["paired_ttest"])
        with pytest.raises(ValueError):
            mod.paired_ttest(np.array([0.5, 0.6]), np.array([0.5]), np.array([0.5, 0.6]))

    @pytest.mark.parametrize("metric_module", ["jsd", "kl"])
    def test_all_identical_returns_pvalue_one(self, metric_module):
        """When all distributions are identical, p-value should be 1.0 (not NaN)."""
        probs = np.full(50, 0.5)
        mod = __import__(f"metrics.{metric_module}", fromlist=["paired_ttest"])
        result = mod.paired_ttest(probs, probs, probs)
        assert not np.isnan(result["p_value"])
        assert result["p_value"] == 1.0

    @pytest.mark.parametrize("metric_module", ["jsd", "kl"])
    def test_significant_when_pert_differs(self, metric_module):
        """Detects when perturbation shifts distributions but baseline doesn't."""
        rng = np.random.default_rng(42)
        orig = rng.uniform(0.4, 0.6, size=200)
        # Perturbation shifts probabilities substantially
        pert = np.clip(orig + rng.uniform(0.2, 0.4, size=200), 0.01, 0.99)
        # Baseline barely changes
        base = np.clip(orig + rng.normal(0, 0.01, size=200), 0.01, 0.99)
        mod = __import__(f"metrics.{metric_module}", fromlist=["paired_ttest"])
        result = mod.paired_ttest(orig, pert, base)
        assert result["p_value"] < 0.05
