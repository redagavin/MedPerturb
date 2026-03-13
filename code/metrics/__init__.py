# ABOUTME: Per-population and per-sample metric modules for MedPerturb
# ABOUTME: Shared validation helpers and bootstrap test logic

import numpy as np


def validate_binary_vectors(x, y):
    """Validate inputs are non-empty equal-length arrays of 0s and 1s."""
    if len(x) == 0:
        raise ValueError("Input arrays must not be empty")
    if len(x) != len(y):
        raise ValueError(f"Array lengths must match: {len(x)} != {len(y)}")
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    vals = np.unique(np.concatenate([x_arr, y_arr]))
    if not np.all(np.isin(vals, [0, 1])):
        raise ValueError(f"Arrays must contain only 0 and 1, got values: {vals}")


def validate_prob_arrays(orig, pert, base):
    """Validate probability arrays are non-empty, equal length, and in [0, 1]."""
    if len(orig) == 0:
        raise ValueError("Input arrays must not be empty")
    if len(orig) != len(pert) or len(orig) != len(base):
        raise ValueError(
            f"Array lengths must match: {len(orig)}, {len(pert)}, {len(base)}"
        )
    for name, arr in [("orig", orig), ("pert", pert), ("base", base)]:
        a = np.asarray(arr)
        if np.any(a < 0) or np.any(a > 1):
            raise ValueError(
                f"Probabilities must be in [0, 1], {name} has range "
                f"[{a.min()}, {a.max()}]"
            )


def bootstrap_test_common(compute_fn, orig, pert, base, n_bootstrap=1000, seed=42):
    """Bootstrap percentile test comparing compute_fn(orig,pert) vs compute_fn(orig,base).

    Two-tailed test with fixed random seed. P-value = proportion of bootstrap
    resamples where the difference falls on the opposite side of zero from the
    observed difference, doubled for two-tailed test.
    """
    rng = np.random.default_rng(seed)
    orig = np.asarray(orig)
    pert = np.asarray(pert)
    base = np.asarray(base)
    n = len(orig)

    observed_pert = compute_fn(orig, pert)
    observed_base = compute_fn(orig, base)
    observed_diff = observed_pert - observed_base

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        diffs[i] = compute_fn(orig[idx], pert[idx]) - compute_fn(orig[idx], base[idx])

    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)

    if observed_diff >= 0:
        p_value = 2 * np.mean(diffs <= 0)
    else:
        p_value = 2 * np.mean(diffs >= 0)
    p_value = min(p_value, 1.0)

    return {
        "statistic_perturbation": observed_pert,
        "statistic_baseline": observed_base,
        "observed_diff": observed_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }
