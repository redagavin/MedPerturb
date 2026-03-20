# Simulation v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the Monte Carlo power simulation to use real MedPerturb logits instead of a synthetic DGP, sweeping perturbation variance (sigma_pert) and baseline noise (sigma) across 6 (question, model) conditions.

**Architecture:** Replace `generate_responses()` with a version that takes real logit arrays z_i and y_orig instead of synthetic beta parameters. Replace the (beta1 × beta_gender × sigma) parameter grid with (sigma_pert × sigma) per condition. Keep the existing multiprocessing/checkpointing infrastructure and the same 5 metric test functions.

**Tech Stack:** Python, numpy, scipy, pandas, matplotlib; existing `code/metrics/` module (unchanged); SLURM for execution.

**Design doc:** `docs/plans/2026-03-19-simulation-v2-design.md`

---

### Task 1: Load experiment logits

Extract z_i and y_orig arrays from MedPerturb experiment JSON files.

**Files:**
- Create: `code/load_experiment_logits.py`
- Test: `tests/unit/test_load_experiment_logits.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_load_experiment_logits.py
# ABOUTME: Tests for loading real logits from MedPerturb experiment results
# ABOUTME: Validates clamping, majority vote, and condition extraction

import json
import math
import numpy as np
import pytest
import tempfile
import os


def _make_fake_results(n=5, p_yes_values=None):
    """Create minimal fake experiment JSON matching real format."""
    entries = []
    for i in range(n):
        entry = {"context_id": i}
        for q in ["MANAGE", "VISIT", "RESOURCE"]:
            p = p_yes_values[i] if p_yes_values else 0.5
            entry[f"original_{q}"] = {
                "logit_probs": p,
                "binary_answers": [1, 1, 0] if p > 0.5 else [0, 0, 1],
            }
        entries.append(entry)
    return entries


class TestLoadCondition:
    """Tests for load_condition extracting z_i and y_orig arrays."""

    def test_returns_z_and_y_orig(self):
        """load_condition returns dict with z_i (logits) and y_orig (binary answers)."""
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(5, [0.3, 0.5, 0.7, 0.2, 0.8]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            assert "z_i" in result
            assert "y_orig" in result
            assert len(result["z_i"]) == 5
            assert len(result["y_orig"]) == 5
        finally:
            os.unlink(path)

    def test_logit_conversion(self):
        """P(Yes) is correctly converted to logit: log(p / (1-p))."""
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(1, [0.7]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            expected = math.log(0.7 / 0.3)
            assert abs(result["z_i"][0] - expected) < 1e-10
        finally:
            os.unlink(path)

    def test_clamps_p_equals_1(self):
        """P(Yes) = 1.0 is clamped to 1 - 1e-6 before logit conversion."""
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(1, [1.0]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            expected = math.log((1.0 - 1e-6) / 1e-6)
            assert abs(result["z_i"][0] - expected) < 1e-6
            assert np.isfinite(result["z_i"][0])
        finally:
            os.unlink(path)

    def test_clamps_p_equals_0(self):
        """P(Yes) = 0.0 is clamped to 1e-6 before logit conversion."""
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(1, [0.0]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            expected = math.log(1e-6 / (1.0 - 1e-6))
            assert abs(result["z_i"][0] - expected) < 1e-6
            assert np.isfinite(result["z_i"][0])
        finally:
            os.unlink(path)

    def test_majority_vote_binary_answers(self):
        """y_orig is majority vote of 3 binary_answers."""
        from load_experiment_logits import load_condition
        entries = [{"context_id": 0, "original_MANAGE": {
            "logit_probs": 0.6,
            "binary_answers": [1, 0, 1],  # majority = 1
        }}]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(entries, f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            assert result["y_orig"][0] == 1
        finally:
            os.unlink(path)

    def test_different_questions(self):
        """Different question names extract from different keys."""
        from load_experiment_logits import load_condition
        entries = [{
            "context_id": 0,
            "original_MANAGE": {"logit_probs": 0.3, "binary_answers": [0, 0, 0]},
            "original_VISIT": {"logit_probs": 0.8, "binary_answers": [1, 1, 1]},
            "original_RESOURCE": {"logit_probs": 0.5, "binary_answers": [0, 1, 0]},
        }]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(entries, f)
            path = f.name
        try:
            m = load_condition(path, "MANAGE")
            v = load_condition(path, "VISIT")
            r = load_condition(path, "RESOURCE")
            assert m["y_orig"][0] == 0
            assert v["y_orig"][0] == 1
            assert r["y_orig"][0] == 0
        finally:
            os.unlink(path)

    def test_z_i_is_numpy_float_array(self):
        """z_i is a numpy float64 array."""
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(3, [0.3, 0.5, 0.7]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            assert isinstance(result["z_i"], np.ndarray)
            assert result["z_i"].dtype == np.float64
        finally:
            os.unlink(path)

    def test_y_orig_is_numpy_int_array(self):
        """y_orig is a numpy integer array of 0s and 1s."""
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(3, [0.3, 0.5, 0.7]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            assert isinstance(result["y_orig"], np.ndarray)
            assert set(np.unique(result["y_orig"])).issubset({0, 1})
        finally:
            os.unlink(path)
```

**Step 2: Run tests to verify they fail**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_load_experiment_logits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'load_experiment_logits'`

**Step 3: Write minimal implementation**

```python
# code/load_experiment_logits.py
# ABOUTME: Loads real logits and binary answers from MedPerturb experiment JSON files.
# ABOUTME: Converts P(Yes) to logits with clamping for extreme probabilities.

import json
import math

import numpy as np


CLAMP_EPSILON = 1e-6


def load_condition(json_path: str, question: str) -> dict:
    """Load z_i (logits) and y_orig (majority-vote binary answers) for one condition.

    Args:
        json_path: Path to experiment result JSON (list of dicts).
        question: One of "MANAGE", "VISIT", "RESOURCE".

    Returns:
        dict with:
            z_i: numpy float64 array of logits, log(p / (1-p))
            y_orig: numpy int array of majority-vote binary answers
    """
    with open(json_path) as f:
        data = json.load(f)

    key = f"original_{question}"
    z_i_list = []
    y_orig_list = []

    for entry in data:
        record = entry[key]

        p = record["logit_probs"]
        p_clamped = max(CLAMP_EPSILON, min(1.0 - CLAMP_EPSILON, p))
        z_i_list.append(math.log(p_clamped / (1.0 - p_clamped)))

        votes = record["binary_answers"]
        y_orig_list.append(1 if sum(votes) >= 2 else 0)

    return {
        "z_i": np.array(z_i_list, dtype=np.float64),
        "y_orig": np.array(y_orig_list, dtype=np.int64),
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_load_experiment_logits.py -v`
Expected: All 8 PASS

**Step 5: Commit**

```bash
git add code/load_experiment_logits.py tests/unit/test_load_experiment_logits.py
git commit -m "feat: add load_experiment_logits for simulation v2 data loading"
```

---

### Task 2: New DGP function

Replace `generate_responses` with a version that takes real logits z_i and y_orig.

**Files:**
- Modify: `code/simulation.py` (add `generate_responses_v2`, keep v1 functions for now)
- Test: `tests/unit/test_simulation.py` (add new test class)

**Step 1: Write failing tests**

Add the following to `tests/unit/test_simulation.py`:

```python
class TestGenerateResponsesV2:
    """Tests for generate_responses_v2 using real logits."""

    def _make_z_and_y(self, n=100, seed=42):
        """Create realistic z_i and y_orig arrays."""
        rng = np.random.default_rng(seed)
        z_i = rng.normal(0, 2, size=n)
        y_orig = (z_i > 0).astype(int)
        return z_i, y_orig

    def test_returns_required_keys(self):
        """Returns dict with orig, pert, base, p_orig, p_pert, p_base."""
        from simulation import generate_responses_v2
        z_i, y_orig = self._make_z_and_y()
        rng = np.random.default_rng(42)
        result = generate_responses_v2(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        assert set(result.keys()) == {"orig", "pert", "base", "p_orig", "p_pert", "p_base"}

    def test_lengths_match_input(self):
        """All output arrays have same length as input z_i."""
        from simulation import generate_responses_v2
        z_i, y_orig = self._make_z_and_y(n=50)
        rng = np.random.default_rng(42)
        result = generate_responses_v2(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        for key in result:
            assert len(result[key]) == 50

    def test_p_orig_is_sigmoid_of_z(self):
        """p_orig = sigmoid(z_i), fixed from real data."""
        from simulation import generate_responses_v2
        from scipy.special import expit as sigmoid
        z_i = np.array([0.0, 1.0, -1.0, 3.0])
        y_orig = np.array([0, 1, 0, 1])
        rng = np.random.default_rng(42)
        result = generate_responses_v2(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        np.testing.assert_array_almost_equal(result["p_orig"], sigmoid(z_i))

    def test_orig_is_y_orig_not_resampled(self):
        """orig binary vector is exactly y_orig (fixed from data, not resampled)."""
        from simulation import generate_responses_v2
        z_i = np.array([0.0, 1.0, -1.0, 3.0])
        y_orig = np.array([0, 1, 0, 1])
        rng = np.random.default_rng(42)
        result = generate_responses_v2(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        np.testing.assert_array_equal(result["orig"], y_orig)

    def test_prob_vectors_in_0_1(self):
        """Probability vectors are in (0, 1)."""
        from simulation import generate_responses_v2
        z_i, y_orig = self._make_z_and_y()
        rng = np.random.default_rng(42)
        result = generate_responses_v2(z_i, y_orig, sigma_pert=1.0, sigma=0.5, rng=rng)
        for key in ["p_orig", "p_pert", "p_base"]:
            assert np.all(result[key] > 0.0)
            assert np.all(result[key] < 1.0)

    def test_binary_vectors_are_binary(self):
        """pert and base binary vectors contain only 0s and 1s."""
        from simulation import generate_responses_v2
        z_i, y_orig = self._make_z_and_y()
        rng = np.random.default_rng(42)
        result = generate_responses_v2(z_i, y_orig, sigma_pert=0.5, sigma=0.3, rng=rng)
        for key in ["pert", "base"]:
            assert set(np.unique(result[key])).issubset({0, 1})

    def test_null_hypothesis_symmetric(self):
        """When sigma_pert=0, pert and base arms have same noise distribution."""
        from simulation import generate_responses_v2
        z_i, y_orig = self._make_z_and_y(n=10000)
        rng = np.random.default_rng(42)
        result = generate_responses_v2(z_i, y_orig, sigma_pert=0.0, sigma=0.5, rng=rng)
        pert_diff = np.mean(np.abs(result["p_pert"] - result["p_orig"]))
        base_diff = np.mean(np.abs(result["p_base"] - result["p_orig"]))
        assert abs(pert_diff - base_diff) < 0.01

    def test_sigma_pert_adds_extra_noise_to_pert_arm(self):
        """When sigma_pert > 0, pert arm has more divergence from original than base."""
        from simulation import generate_responses_v2
        z_i, y_orig = self._make_z_and_y(n=10000)
        rng = np.random.default_rng(42)
        result = generate_responses_v2(z_i, y_orig, sigma_pert=1.0, sigma=0.3, rng=rng)
        pert_diff = np.mean(np.abs(result["p_pert"] - result["p_orig"]))
        base_diff = np.mean(np.abs(result["p_base"] - result["p_orig"]))
        assert pert_diff > base_diff + 0.01

    def test_zero_noise_preserves_original(self):
        """When sigma_pert=0 and sigma=0, p_pert = p_base = p_orig."""
        from simulation import generate_responses_v2
        z_i = np.array([0.0, 1.0, -1.0, 3.0])
        y_orig = np.array([0, 1, 0, 1])
        rng = np.random.default_rng(42)
        result = generate_responses_v2(z_i, y_orig, sigma_pert=0.0, sigma=0.0, rng=rng)
        np.testing.assert_array_almost_equal(result["p_pert"], result["p_orig"])
        np.testing.assert_array_almost_equal(result["p_base"], result["p_orig"])

    def test_deterministic_with_same_seed(self):
        """Same RNG produces identical output."""
        from simulation import generate_responses_v2
        z_i, y_orig = self._make_z_and_y()
        r1 = generate_responses_v2(z_i, y_orig, 0.5, 0.3, np.random.default_rng(42))
        r2 = generate_responses_v2(z_i, y_orig, 0.5, 0.3, np.random.default_rng(42))
        for key in result_keys:
            np.testing.assert_array_equal(r1[key], r2[key])
```

**Step 2: Run tests to verify they fail**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py::TestGenerateResponsesV2 -v`
Expected: FAIL — `ImportError: cannot import name 'generate_responses_v2'`

**Step 3: Write minimal implementation**

Add to `code/simulation.py` after the existing `generate_responses` function:

```python
def generate_responses_v2(
    z_i: np.ndarray,
    y_orig: np.ndarray,
    sigma_pert: float,
    sigma: float,
    rng: np.random.Generator,
) -> dict:
    """Generate simulated responses from real logits.

    For each case i with real logit z_i:
        epsilon_pert_i  ~ N(0, sigma_pert^2)
        epsilon_noise_i ~ N(0, sigma^2)       (perturbation arm)
        epsilon_noise_i'~ N(0, sigma^2)       (baseline arm, independent)

        logit_pert_i = z_i + epsilon_pert_i + epsilon_noise_i
        logit_base_i = z_i + epsilon_noise_i'

        p_pert = sigmoid(logit_pert), p_base = sigmoid(logit_base)
        y_pert ~ Bernoulli(p_pert), y_base ~ Bernoulli(p_base)

    Returns dict with binary vectors and probability vectors.
    """
    n = len(z_i)

    epsilon_pert = rng.normal(0, sigma_pert, size=n) if sigma_pert > 0 else 0.0
    epsilon_noise_pert = rng.normal(0, sigma, size=n) if sigma > 0 else 0.0
    epsilon_noise_base = rng.normal(0, sigma, size=n) if sigma > 0 else 0.0

    logit_pert = z_i + epsilon_pert + epsilon_noise_pert
    logit_base = z_i + epsilon_noise_base

    p_orig = sigmoid(z_i)
    p_pert = sigmoid(logit_pert)
    p_base = sigmoid(logit_base)

    pert_bin = rng.binomial(1, p_pert)
    base_bin = rng.binomial(1, p_base)

    return {
        "orig": y_orig.copy(),
        "pert": pert_bin,
        "base": base_bin,
        "p_orig": p_orig,
        "p_pert": p_pert,
        "p_base": p_base,
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py::TestGenerateResponsesV2 -v`
Expected: All 11 PASS

**Step 5: Commit**

```bash
git add code/simulation.py tests/unit/test_simulation.py
git commit -m "feat: add generate_responses_v2 using real logits"
```

---

### Task 3: Combo seed and runner for v2

New _combo_seed_v2 with formatted float strings, and _run_one_combo_v2 that accepts real logits.

**Files:**
- Modify: `code/simulation.py`
- Test: `tests/unit/test_simulation.py`

**Step 1: Write failing tests**

Add to `tests/unit/test_simulation.py`:

```python
class TestComboSeedV2:
    """Tests for deterministic per-combo seeding with formatted floats."""

    def test_same_params_same_seed(self):
        """Identical parameters produce identical seeds."""
        from simulation import _combo_seed_v2
        s1 = _combo_seed_v2(42, 0.5, 0.25, "MANAGE_8b")
        s2 = _combo_seed_v2(42, 0.5, 0.25, "MANAGE_8b")
        assert s1 == s2

    def test_different_params_different_seed(self):
        """Different parameters produce different seeds."""
        from simulation import _combo_seed_v2
        s1 = _combo_seed_v2(42, 0.5, 0.25, "MANAGE_8b")
        s2 = _combo_seed_v2(42, 0.6, 0.25, "MANAGE_8b")
        assert s1 != s2

    def test_different_conditions_different_seed(self):
        """Different conditions produce different seeds."""
        from simulation import _combo_seed_v2
        s1 = _combo_seed_v2(42, 0.5, 0.25, "MANAGE_8b")
        s2 = _combo_seed_v2(42, 0.5, 0.25, "VISIT_8b")
        assert s1 != s2

    def test_float_formatting_canonical(self):
        """Floating-point artifacts don't change the seed (uses formatted strings)."""
        from simulation import _combo_seed_v2
        # 0.3 from integer division vs np.arange artifact
        s1 = _combo_seed_v2(42, 0.3, 0.25, "MANAGE_8b")
        s2 = _combo_seed_v2(42, 0.30000000000000004, 0.25, "MANAGE_8b")
        assert s1 == s2

    def test_seed_in_valid_range(self):
        """Seed is a non-negative integer below 2^31."""
        from simulation import _combo_seed_v2
        s = _combo_seed_v2(42, 0.5, 0.25, "MANAGE_8b")
        assert isinstance(s, int)
        assert 0 <= s < 2**31
```

**Step 2: Run tests to verify they fail**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py::TestComboSeedV2 -v`
Expected: FAIL — `ImportError: cannot import name '_combo_seed_v2'`

**Step 3: Write minimal implementation**

Add to `code/simulation.py` after `_combo_seed`:

```python
def _combo_seed_v2(global_seed, sigma_pert, sigma, condition):
    """Derive a deterministic seed from v2 parameters.

    Uses formatted float strings (:.1f for sigma_pert, :.2f for sigma) to ensure
    floating-point representation artifacts don't produce different seeds.
    """
    import hashlib
    key = f"{global_seed}:{sigma_pert:.1f}:{sigma:.2f}:{condition}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def _run_one_combo_v2(args):
    """Run all simulations for one (sigma_pert, sigma) combo using real logits.

    Designed as a top-level function for multiprocessing.Pool.
    """
    sigma_pert, sigma, condition, n_simulations, n_bootstrap, global_seed, z_i, y_orig = args

    combo_seed = _combo_seed_v2(global_seed, sigma_pert, sigma, condition)
    base_rng = np.random.default_rng(combo_seed)

    p_values_by_metric = {m: [] for m in ALL_METRICS}

    for _ in range(n_simulations):
        sim_seed = base_rng.integers(0, 2**31)
        sim_rng = np.random.default_rng(sim_seed)

        data = generate_responses_v2(z_i, y_orig, sigma_pert, sigma, sim_rng)
        boot_rng = np.random.default_rng(sim_rng.integers(0, 2**31))
        pvals = run_single_simulation(data, n_bootstrap=n_bootstrap, rng=boot_rng)
        for m in ALL_METRICS:
            p_values_by_metric[m].append(pvals[m])

    results = []
    for m in ALL_METRICS:
        pv = np.array(p_values_by_metric[m])
        results.append({
            "metric": m,
            "sigma_pert": sigma_pert,
            "sigma": sigma,
            "condition": condition,
            "detection_rate": float(np.mean(pv < 0.05)),
            "mean_p_value": float(np.mean(pv)),
        })
    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py::TestComboSeedV2 -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
git add code/simulation.py tests/unit/test_simulation.py
git commit -m "feat: add v2 combo seed and runner with formatted float keys"
```

---

### Task 4: Power analysis loop for v2

New `run_power_analysis_v2` with the (sigma_pert × sigma) grid and per-condition checkpointing.

**Files:**
- Modify: `code/simulation.py`
- Test: `tests/unit/test_simulation.py`

**Step 1: Write failing tests**

Add to `tests/unit/test_simulation.py`:

```python
class TestRunPowerAnalysisV2:
    """Tests for the v2 power analysis loop with real logits."""

    def _make_z_and_y(self, n=50, seed=42):
        rng = np.random.default_rng(seed)
        z_i = rng.normal(0, 1.5, size=n)
        y_orig = (z_i > 0).astype(int)
        return z_i, y_orig

    def test_output_columns(self):
        """Output DataFrame has v2 columns (sigma_pert, sigma, condition)."""
        from simulation import run_power_analysis_v2
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis_v2(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 0.5], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
        )
        required = {"metric", "sigma_pert", "sigma", "condition", "detection_rate", "mean_p_value"}
        assert required == set(results.columns)

    def test_output_row_count(self):
        """One row per (metric, sigma_pert, sigma) combination."""
        from simulation import run_power_analysis_v2
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis_v2(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 0.5, 1.0], sigma_values=[0.25, 0.5],
            n_simulations=5, n_bootstrap=50, seed=42,
        )
        # 5 metrics x 3 sigma_pert x 2 sigma = 30 rows
        assert len(results) == 30

    def test_all_five_metrics(self):
        """Results include all 5 metrics."""
        from simulation import run_power_analysis_v2
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis_v2(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
        )
        assert set(results["metric"].unique()) == {"mi", "phi", "flip_rate", "jsd", "kl"}

    def test_detection_rate_between_0_and_1(self):
        """Detection rates are valid proportions."""
        from simulation import run_power_analysis_v2
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis_v2(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 1.0], sigma_values=[0.3],
            n_simulations=10, n_bootstrap=50, seed=42,
        )
        assert (results["detection_rate"] >= 0).all()
        assert (results["detection_rate"] <= 1).all()

    def test_condition_column_populated(self):
        """All rows have the correct condition string."""
        from simulation import run_power_analysis_v2
        z_i, y_orig = self._make_z_and_y()
        results = run_power_analysis_v2(
            z_i=z_i, y_orig=y_orig, condition="VISIT_70b",
            sigma_pert_values=[0.0], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
        )
        assert (results["condition"] == "VISIT_70b").all()

    def test_null_calibration(self):
        """Under H0 (sigma_pert=0, sigma>0), detection rate is below 0.20."""
        from simulation import run_power_analysis_v2
        z_i, y_orig = self._make_z_and_y(n=100)
        results = run_power_analysis_v2(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0], sigma_values=[0.5],
            n_simulations=200, n_bootstrap=200, seed=42,
        )
        for metric in ["mi", "phi", "flip_rate", "jsd", "kl"]:
            rate = results[results["metric"] == metric]["detection_rate"].values[0]
            assert rate < 0.20, f"{metric} null rate {rate} too high"

    def test_checkpoint_and_resume(self, tmp_path):
        """Checkpointing works: partial run + resume produces same result as full run."""
        from simulation import run_power_analysis_v2
        z_i, y_orig = self._make_z_and_y()
        checkpoint = str(tmp_path / "ckpt.csv")

        # Full run
        full = run_power_analysis_v2(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 0.5, 1.0], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
        )

        # Simulate partial checkpoint (first combo only)
        partial_rows = full[
            (full["sigma_pert"] == 0.0) & (full["sigma"] == 0.3)
        ].copy()
        partial_rows.to_csv(checkpoint, index=False)

        # Resume from checkpoint
        resumed = run_power_analysis_v2(
            z_i=z_i, y_orig=y_orig, condition="MANAGE_8b",
            sigma_pert_values=[0.0, 0.5, 1.0], sigma_values=[0.3],
            n_simulations=5, n_bootstrap=50, seed=42,
            checkpoint_path=checkpoint,
        )

        assert len(resumed) == len(full)
        # Checkpoint rows should be unchanged
        for _, row in partial_rows.iterrows():
            match = resumed[
                (resumed["metric"] == row["metric"]) &
                (resumed["sigma_pert"] == row["sigma_pert"]) &
                (resumed["sigma"] == row["sigma"])
            ]
            assert len(match) == 1
            assert match["detection_rate"].values[0] == row["detection_rate"]
```

**Step 2: Run tests to verify they fail**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py::TestRunPowerAnalysisV2 -v`
Expected: FAIL — `ImportError: cannot import name 'run_power_analysis_v2'`

**Step 3: Write minimal implementation**

Add to `code/simulation.py`:

```python
def run_power_analysis_v2(
    z_i: np.ndarray,
    y_orig: np.ndarray,
    condition: str,
    sigma_pert_values: list[float],
    sigma_values: list[float],
    n_simulations: int = 1000,
    n_bootstrap: int = 1000,
    seed: int = 42,
    n_workers: int = 1,
    checkpoint_path: str = None,
) -> pd.DataFrame:
    """Run Monte Carlo simulation across (sigma_pert, sigma) grid for one condition.

    Uses real logits z_i and binary answers y_orig. Parallelizes across
    parameter combos using n_workers processes. Checkpoints to CSV.

    Returns DataFrame with columns: metric, sigma_pert, sigma, condition,
    detection_rate, mean_p_value.
    """
    import multiprocessing

    # Load checkpoint if exists
    completed = set()
    all_results = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        all_results = existing.to_dict('records')
        for _, row in existing.iterrows():
            completed.add((row['sigma_pert'], row['sigma'], row['metric']))
        n_done = len(completed) // len(ALL_METRICS)
        print(f"  Resuming from checkpoint: {n_done} combos already done", flush=True)

    # Build work items
    work_items = []
    for sigma_pert in sigma_pert_values:
        for sigma in sigma_values:
            if (sigma_pert, sigma, ALL_METRICS[0]) in completed:
                continue
            work_items.append((
                sigma_pert, sigma, condition,
                n_simulations, n_bootstrap, seed, z_i, y_orig,
            ))

    total = len(work_items)
    print(f"  {total} combos remaining, using {n_workers} workers", flush=True)

    if total == 0:
        return pd.DataFrame(all_results)

    done = 0
    if n_workers <= 1:
        for item in work_items:
            combo_results = _run_one_combo_v2(item)
            all_results.extend(combo_results)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)", flush=True)
                if checkpoint_path:
                    pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)
    else:
        with multiprocessing.Pool(n_workers) as pool:
            for combo_results in pool.imap_unordered(_run_one_combo_v2, work_items):
                all_results.extend(combo_results)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)", flush=True)
                    if checkpoint_path:
                        pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    if checkpoint_path:
        pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(all_results)
```

**Step 4: Run tests to verify they pass**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py::TestRunPowerAnalysisV2 -v`
Expected: All 8 PASS

**Step 5: Commit**

```bash
git add code/simulation.py tests/unit/test_simulation.py
git commit -m "feat: add run_power_analysis_v2 with sigma_pert x sigma grid"
```

---

### Task 5: Power curve generation for v2

New figure layout: 4 subplots (one per sigma), 5 metric curves per subplot.

**Files:**
- Modify: `code/simulation.py`
- Test: `tests/unit/test_simulation.py`

**Step 1: Write failing tests**

Add to `tests/unit/test_simulation.py`:

```python
class TestGeneratePowerCurvesV2:
    """Tests for v2 power curve figure generation."""

    def test_creates_png_file(self, tmp_path):
        """Generates a PNG file for the condition."""
        from simulation import generate_power_curves_v2
        # Minimal results DataFrame
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
        generate_power_curves_v2(results, str(tmp_path), "MANAGE_8b")
        assert (tmp_path / "power_curves_MANAGE_8b.png").exists()

    def test_no_error_on_single_sigma(self, tmp_path):
        """Works with a single sigma value."""
        from simulation import generate_power_curves_v2
        rows = []
        for metric in ["mi", "phi", "flip_rate", "jsd", "kl"]:
            for sp in [0.0, 0.5]:
                rows.append({
                    "metric": metric, "sigma_pert": sp, "sigma": 0.5,
                    "condition": "MANAGE_8b", "detection_rate": 0.05,
                    "mean_p_value": 0.5,
                })
        results = pd.DataFrame(rows)
        generate_power_curves_v2(results, str(tmp_path), "MANAGE_8b")
        assert (tmp_path / "power_curves_MANAGE_8b.png").exists()
```

**Step 2: Run tests to verify they fail**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py::TestGeneratePowerCurvesV2 -v`
Expected: FAIL — `ImportError: cannot import name 'generate_power_curves_v2'`

**Step 3: Write minimal implementation**

Add to `code/simulation.py`:

```python
def generate_power_curves_v2(results: pd.DataFrame, output_dir: str, condition: str) -> None:
    """Generate power curve figure for one condition.

    Layout: one subplot per sigma value, 5 metric curves per subplot.
    """
    sigma_values = sorted(results["sigma"].unique())
    n_cols = min(len(sigma_values), 2)
    n_rows = (len(sigma_values) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    metric_colors = {
        "jsd": "#1f77b4", "kl": "#ff7f0e",
        "mi": "#2ca02c", "phi": "#d62728", "flip_rate": "#9467bd",
    }

    for idx, sigma in enumerate(sigma_values):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        subset = results[results["sigma"] == sigma]

        for metric in ["jsd", "kl", "mi", "phi", "flip_rate"]:
            m_data = subset[subset["metric"] == metric].sort_values("sigma_pert")
            ax.plot(
                m_data["sigma_pert"], m_data["detection_rate"],
                marker='o', markersize=3, color=metric_colors[metric],
                label=metric.upper() if metric != "flip_rate" else "Flip Rate",
            )

        ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=0.80, color='lightgray', linestyle=':', linewidth=1)
        ax.set_xlabel(r'$\sigma_{pert}$')
        ax.set_ylabel('Detection rate')
        ax.set_title(f'{condition} ($\\sigma$={sigma})')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(sigma_values), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'power_curves_{condition}.png'), dpi=300)
    plt.close(fig)
```

**Step 4: Run tests to verify they pass**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py::TestGeneratePowerCurvesV2 -v`
Expected: All 2 PASS

**Step 5: Commit**

```bash
git add code/simulation.py tests/unit/test_simulation.py
git commit -m "feat: add v2 power curve generation with per-sigma subplots"
```

---

### Task 6: Main CLI and SLURM script

Wire everything together: CLI arg parsing, condition loop, SLURM sbatch.

**Files:**
- Modify: `code/simulation.py` (add `main_v2`)
- Modify: `slurm/run_simulation.sbatch`
- Test: manual smoke test (CLI --help, dry run with small params)

**Step 1: Write `main_v2` function**

Add to `code/simulation.py`, replacing or renaming the existing `main`:

```python
CONDITIONS = [
    ("MANAGE", "8b", "results/main_evaluation_llama_3.1_8b_instruct.json"),
    ("VISIT", "8b", "results/main_evaluation_llama_3.1_8b_instruct.json"),
    ("RESOURCE", "8b", "results/main_evaluation_llama_3.1_8b_instruct.json"),
    ("MANAGE", "70b", "results/main_evaluation_llama_3.1_70b_instruct.json"),
    ("VISIT", "70b", "results/main_evaluation_llama_3.1_70b_instruct.json"),
    ("RESOURCE", "70b", "results/main_evaluation_llama_3.1_70b_instruct.json"),
]


def main():
    import argparse
    from load_experiment_logits import load_condition

    parser = argparse.ArgumentParser(
        description="Monte Carlo power simulation v2 — empirically grounded"
    )
    parser.add_argument('--sigma-pert-max', type=float, default=3.0,
                        help='Maximum sigma_pert to sweep')
    parser.add_argument('--sigma-pert-step', type=float, default=0.1,
                        help='Step size for sigma_pert sweep')
    parser.add_argument('--sigma-values', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 1.0],
                        help='Baseline noise levels')
    parser.add_argument('--n-simulations', type=int, default=1000)
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='results/simulation_v2')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_steps = int(round(args.sigma_pert_max / args.sigma_pert_step)) + 1
    sigma_pert_values = [i * args.sigma_pert_step for i in range(n_steps)]

    print("Running power analysis v2 (empirically grounded):", flush=True)
    print(f"  sigma_pert: 0 to {args.sigma_pert_max} (step {args.sigma_pert_step},"
          f" {len(sigma_pert_values)} values)", flush=True)
    print(f"  sigma: {args.sigma_values}", flush=True)
    print(f"  {args.n_simulations} simulations, {args.n_bootstrap} bootstrap", flush=True)
    print(f"  Workers: {args.n_workers}", flush=True)
    print(f"  Conditions: {len(CONDITIONS)}", flush=True)

    for question, model_short, json_path in CONDITIONS:
        condition = f"{question}_{model_short}"
        print(f"\n{'='*50}", flush=True)
        print(f"Condition: {condition}", flush=True)
        print(f"{'='*50}", flush=True)

        data = load_condition(json_path, question)
        print(f"  Loaded {len(data['z_i'])} cases, "
              f"z_i range: [{data['z_i'].min():.1f}, {data['z_i'].max():.1f}]", flush=True)

        checkpoint = os.path.join(args.output_dir, f'simulation_v2_{condition}.csv')
        results = run_power_analysis_v2(
            z_i=data["z_i"], y_orig=data["y_orig"], condition=condition,
            sigma_pert_values=sigma_pert_values, sigma_values=args.sigma_values,
            n_simulations=args.n_simulations, n_bootstrap=args.n_bootstrap,
            seed=args.seed, n_workers=args.n_workers,
            checkpoint_path=checkpoint,
        )

        results.to_csv(checkpoint, index=False)
        print(f"  Results saved to: {checkpoint}", flush=True)

        generate_power_curves_v2(results, args.output_dir, condition)
        print(f"  Power curves saved", flush=True)

    print(f"\nAll conditions complete.", flush=True)
```

**Step 2: Update the ABOUTME comments**

Update lines 1-2 of `code/simulation.py`:

```python
# ABOUTME: Monte Carlo power simulation for all 5 MedPerturb metrics.
# ABOUTME: v2: uses real experiment logits instead of synthetic DGP.
```

**Step 3: Update SLURM script**

Replace `slurm/run_simulation.sbatch` content:

```bash
#!/bin/bash
#SBATCH --job-name=simulation_v2
#SBATCH --partition=177huntington
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/simulation_%j.out
#SBATCH --error=logs/simulation_%j.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for Monte Carlo power simulation v2 (empirically grounded)
# ABOUTME: CPU-only job, iterates over 6 (question, model) conditions

echo "========================================"
echo "Power Simulation v2 (Empirically Grounded)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

source ~/.bashrc
conda activate cot

MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"
cd "${MEDPERTURB_DIR}"

mkdir -p results/simulation_v2

python -u code/simulation.py \
    --n-simulations 1000 \
    --n-bootstrap 1000 \
    --sigma-pert-max 3.0 \
    --sigma-pert-step 0.1 \
    --sigma-values 0.0 0.25 0.5 1.0 \
    --seed 42 \
    --n-workers 32 \
    --output-dir results/simulation_v2

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
```

**Step 4: Smoke test**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot python code/simulation.py --help`
Expected: Shows v2 args (--sigma-pert-max, --sigma-values, etc.)

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot python code/simulation.py --sigma-pert-max 0.1 --sigma-pert-step 0.1 --sigma-values 0.3 --n-simulations 5 --n-bootstrap 50 --output-dir /tmp/sim_v2_test`
Expected: Runs 6 conditions quickly, produces CSV and PNG files in `/tmp/sim_v2_test/`

**Step 5: Run ALL existing + new tests**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py tests/unit/test_load_experiment_logits.py -v`

Note: The v1 tests in `TestGenerateResponses`, `TestRunSingleSimulation`, and `TestRunPowerAnalysis` still test the v1 functions which are preserved in the file. They should still pass. If any fail because `main()` was replaced, adjust the `if __name__ == '__main__'` block to call the new `main()`.

**Step 6: Commit**

```bash
git add code/simulation.py slurm/run_simulation.sbatch
git commit -m "feat: wire up simulation v2 main CLI and SLURM script"
```

---

### Task 7: Clean up v1 code

After v2 is verified working, remove v1-specific code that is no longer reachable from `main()`.

**Files:**
- Modify: `code/simulation.py` (remove v1 functions)
- Modify: `tests/unit/test_simulation.py` (remove v1 test classes)

**Step 1: Identify v1-only code**

Functions to remove from `code/simulation.py`:
- `generate_responses` (replaced by `generate_responses_v2`)
- `_combo_seed` (replaced by `_combo_seed_v2`)
- `_run_one_combo` (replaced by `_run_one_combo_v2`)
- `run_power_analysis` (replaced by `run_power_analysis_v2`)
- `generate_power_curves` (replaced by `generate_power_curves_v2`)

Test classes to remove from `tests/unit/test_simulation.py`:
- `TestGenerateResponses`
- `TestRunPowerAnalysis` (v1 version)

Keep: `TestRunSingleSimulation` (still used by v2)

**Step 2: Remove v1 code and tests**

Remove the identified functions and test classes.

**Step 3: Rename v2 functions (drop the `_v2` suffix)**

Rename in `code/simulation.py`:
- `generate_responses_v2` → `generate_responses`
- `_combo_seed_v2` → `_combo_seed`
- `_run_one_combo_v2` → `_run_one_combo`
- `run_power_analysis_v2` → `run_power_analysis`
- `generate_power_curves_v2` → `generate_power_curves`

Update all references in `main()`, test files, and internal calls.

**Step 4: Run all tests**

Run: `cd /scratch/yang.zih/cot_faithfulness/MedPerturb && conda run -n cot pytest tests/unit/test_simulation.py tests/unit/test_load_experiment_logits.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add code/simulation.py tests/unit/test_simulation.py
git commit -m "refactor: remove v1 simulation code, rename v2 functions"
```

---

### Task 8: End-to-end verification

Run a quick simulation on real data to verify null calibration and basic power properties.

**Step 1: Run quick simulation**

```bash
cd /scratch/yang.zih/cot_faithfulness/MedPerturb
conda run -n cot python code/simulation.py \
    --sigma-pert-max 1.0 --sigma-pert-step 0.5 \
    --sigma-values 0.0 0.5 \
    --n-simulations 200 --n-bootstrap 200 \
    --output-dir results/simulation_v2_quick
```

**Step 2: Verify null calibration**

```bash
conda run -n cot python -c "
import pandas as pd
import glob
files = glob.glob('results/simulation_v2_quick/simulation_v2_*.csv')
for f in sorted(files):
    df = pd.read_csv(f)
    null = df[(df['sigma_pert'] == 0.0) & (df['sigma'] == 0.5)]
    condition = null['condition'].iloc[0]
    print(f'{condition}:')
    for _, row in null.iterrows():
        print(f'  {row[\"metric\"]}: rate={row[\"detection_rate\"]:.3f}')
"
```

Expected: All detection rates at sigma_pert=0 are below 0.15 (with 200 simulations, sampling noise is higher).

**Step 3: Verify power increases with sigma_pert**

```bash
conda run -n cot python -c "
import pandas as pd
import glob
files = glob.glob('results/simulation_v2_quick/simulation_v2_*.csv')
for f in sorted(files):
    df = pd.read_csv(f)
    sig05 = df[df['sigma'] == 0.5]
    condition = sig05['condition'].iloc[0]
    print(f'{condition}:')
    for metric in ['jsd', 'kl']:
        m = sig05[sig05['metric'] == metric].sort_values('sigma_pert')
        rates = m['detection_rate'].values
        print(f'  {metric}: {list(rates)}')
"
```

Expected: JSD/KL detection rates increase with sigma_pert.

**Step 4: Verify figures exist**

```bash
ls results/simulation_v2_quick/power_curves_*.png
```

Expected: 6 PNG files.

**Step 5: Commit quick results**

```bash
git add results/simulation_v2_quick/
git commit -m "test: add quick simulation v2 results for verification"
```

**Step 6: Clean up quick results (optional)**

```bash
rm -rf results/simulation_v2_quick/
```
