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

Both arms now get independent noise terms:

```python
# AFTER (correct)
epsilon_pert = rng.normal(0, sigma, size=n_cases) if sigma > 0 else 0.0
epsilon_base = rng.normal(0, sigma, size=n_cases) if sigma > 0 else 0.0
logit_pert = logit_orig + beta_gender + epsilon_pert
logit_base = logit_orig + epsilon_base
```

Under H0, both arms have the same noise distribution, so paired differences are centered at zero — proper null behavior. The null calibration test was also updated to run at `sigma = 0.3` so it exercises non-degenerate JSD/KL paired differences.

## Lesson

When simulating a controlled experiment with per-sample (continuous) metrics, both arms must be symmetric under H0. The perturbation arm should differ from baseline only by the effect parameter, not by presence/absence of noise.
