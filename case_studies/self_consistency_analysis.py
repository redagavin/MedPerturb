# ABOUTME: Self-consistency baseline analysis for MedPerturb main experiment
# ABOUTME: Computes per-case empirical JSD and aggregate permutation tests

import math


def jsd_bernoulli(p: float, q: float, base: float = 2.0) -> float:
    """Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q).

    Uses 0*log(0) = 0 convention. Bounded in [0, log_base(2)]; base=2 -> max 1.0.

    Boundary safety: the only case where m=0 (or 1-m=0) is when p=q=0 (or p=q=1).
    The p == q short-circuit catches both, so division by zero is impossible
    in the subsequent kl_term calls.
    """
    if p == q:
        return 0.0
    m = (p + q) / 2

    def kl_term(x, y):
        if x == 0:
            return 0.0
        return x * math.log(x / y, base)

    kl_p_m = kl_term(p, m) + kl_term(1 - p, 1 - m)
    kl_q_m = kl_term(q, m) + kl_term(1 - q, 1 - m)
    return (kl_p_m + kl_q_m) / 2
