# Simulation DGP Fix: Asymmetric Noise Bug (2026-03-13)

## Bug

The simulation's data generating process (`simulation.py`) modeled noise asymmetrically between the perturbation and baseline arms:

```python
# BEFORE (broken)
logit_pert = logit_orig + beta_gender          # no noise
logit_base = logit_orig + epsilon              # noise
```

Under H0 (`beta_gender=0`):
- `p_pert = sigmoid(logit_orig)` — exactly equal to `p_orig`
- `p_base = sigmoid(logit_orig + epsilon)` — differs from `p_orig` by noise

This meant `JSD(p_orig, p_pert) = 0` always, while `JSD(p_orig, p_base) > 0`. The paired t-test saw a systematic negative difference and rejected with ~100% probability — a false positive.

## Impact

- **At `sigma > 0`**: 100% false positive rate for JSD and KL (per-sample metrics).
- **At `sigma = 0` (default)**: Both arms had zero noise, so all JSD/KL differences were exactly zero. The test returned `p = 1.0` — not wrong, but degenerate. It didn't actually validate null calibration.
- **Per-population metrics (MI, Phi, Flip Rate)** were unaffected because they consume Bernoulli-sampled binary vectors, which add natural stochastic noise regardless.

No published results were affected because the default configuration uses `sigma = 0`, where the bug is dormant.

## Fix

Added a separate `sigma_pert` parameter (default 0) for perturbation arm noise, independent from `sigma` (baseline noise):

```python
# AFTER (correct)
epsilon_pert = rng.normal(0, sigma_pert, size=n_cases) if sigma_pert > 0 else 0.0
epsilon_base = rng.normal(0, sigma, size=n_cases) if sigma > 0 else 0.0
logit_pert = logit_orig + beta_gender + epsilon_pert
logit_base = logit_orig + epsilon_base
```

### Design rationale

`sigma` models baseline-specific noise from paraphrasing or neutral sentence insertion — variation that is inherent to the baseline transformation, not part of the targeted perturbation. The perturbation (gender swap, age swap) is a targeted, controlled text change that may introduce its own noise, but of a different magnitude.

`sigma_pert` defaults to `sigma` when not specified, ensuring both arms are symmetric under H0 by default. This prevents the original bug from silently reappearing when `sigma > 0`. To model asymmetric noise, set `sigma_pert` explicitly (e.g., `--sigma-pert 0` for no perturbation noise).

## Lesson

When simulating a controlled experiment with per-sample (continuous) metrics, both arms must be symmetric under H0. If the noise parameters differ between arms, JSD/KL will detect the noise asymmetry as a "signal" even when the effect parameter is zero.
