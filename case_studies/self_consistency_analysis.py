# ABOUTME: Self-consistency baseline analysis for MedPerturb main experiment
# ABOUTME: Computes per-case empirical JSD and aggregate permutation tests

import math
import sys
import numpy as np


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


MIN_CLEAN_SAMPLES = 5


def permutation_test_cell(cases, perturbation, task, n_perms, rng,
                          base=2.0, drop_log=None):
    """For one (perturbation, task), compute observed mean JSD across cases plus
    aggregate permutation-test p-value via within-case label shuffling."""
    orig_key = f"original_{task}"
    pert_key = f"{perturbation}_{task}"

    per_case_pools = []
    per_case_observed = []
    n_dropped_missing_key = 0
    n_dropped_empty = 0
    n_dropped_low_count = 0

    for case in cases:
        cid = case.get('context_id', '<unknown>')
        if orig_key not in case or pert_key not in case:
            n_dropped_missing_key += 1
            if drop_log is not None:
                drop_log.append((cid, perturbation, task, 'missing_key'))
            continue
        orig = [x for x in case[orig_key]['binary_answers'] if x in (0, 1)]
        pert = [x for x in case[pert_key]['binary_answers'] if x in (0, 1)]
        if not orig or not pert:
            n_dropped_empty += 1
            if drop_log is not None:
                drop_log.append((cid, perturbation, task, 'empty_after_parse_filter'))
            continue
        if len(orig) < MIN_CLEAN_SAMPLES or len(pert) < MIN_CLEAN_SAMPLES:
            n_dropped_low_count += 1
            if drop_log is not None:
                drop_log.append((cid, perturbation, task,
                                 f'low_count_orig={len(orig)}_pert={len(pert)}'))
            continue
        per_case_observed.append(jsd_bernoulli(
            sum(orig)/len(orig), sum(pert)/len(pert), base=base
        ))
        per_case_pools.append((orig, pert))

    if not per_case_pools:
        print(f"WARNING: no usable cases for ({perturbation}, {task})", file=sys.stderr)
        return {
            'observed_mean_jsd': float('nan'),
            'null_mean_jsd': float('nan'),
            'null_std_jsd': float('nan'),
            'jsd_excess': float('nan'),
            'p_value': float('nan'),
            'n_cases_used': 0,
            'n_active': 0,
            'n_dropped_missing_key': n_dropped_missing_key,
            'n_dropped_empty': n_dropped_empty,
            'n_dropped_low_count': n_dropped_low_count,
        }

    observed_mean = float(np.mean(per_case_observed))

    def _is_active(orig, pert):
        pool = orig + pert
        return not (all(x == 0 for x in pool) or all(x == 1 for x in pool))
    n_active = sum(1 for o, p in per_case_pools if _is_active(o, p))

    null_means = np.empty(n_perms)
    for k in range(n_perms):
        per_case_null = []
        for orig, pert in per_case_pools:
            pool = list(orig) + list(pert)
            n_orig = len(orig)
            rng.shuffle(pool)
            p_emp = sum(pool[:n_orig]) / n_orig
            q_emp = sum(pool[n_orig:]) / (len(pool) - n_orig)
            per_case_null.append(jsd_bernoulli(p_emp, q_emp, base=base))
        null_means[k] = np.mean(per_case_null)

    # Phipson & Smyth (2010) unbiased exact p-value
    p_value = float((1 + (null_means >= observed_mean).sum()) / (1 + n_perms))

    return {
        'observed_mean_jsd': observed_mean,
        'null_mean_jsd': float(null_means.mean()),
        'null_std_jsd': float(null_means.std()),
        'jsd_excess': observed_mean - float(null_means.mean()),
        'p_value': p_value,
        'n_cases_used': len(per_case_pools),
        'n_active': n_active,
        'n_dropped_missing_key': n_dropped_missing_key,
        'n_dropped_empty': n_dropped_empty,
        'n_dropped_low_count': n_dropped_low_count,
    }
