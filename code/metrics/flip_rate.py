# ABOUTME: Flip rate metric between binary answer vectors
# ABOUTME: Proportion of samples where answer changed, with bootstrap test

import numpy as np
from metrics import validate_binary_vectors, bootstrap_test_common


def compute(x, y):
    """Compute flip rate: proportion of positions where x != y."""
    x = np.asarray(x)
    y = np.asarray(y)
    validate_binary_vectors(x, y)
    return np.mean(x != y)


def bootstrap_test(orig, pert, base, n_bootstrap=1000, seed=42):
    """Bootstrap test comparing compute(orig,pert) vs compute(orig,base)."""
    return bootstrap_test_common(compute, orig, pert, base, n_bootstrap, seed)
