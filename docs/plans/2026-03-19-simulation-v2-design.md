# Simulation v2: Empirically-Grounded Power Analysis

**Date:** 2026-03-19
**Goal:** Redesign the Monte Carlo power simulation to use real model logits from MedPerturb experiments instead of a fully synthetic data generating process.

---

## Motivation

The v1 simulation used a fully synthetic DGP with parameters (beta0, beta1, beta_gender, sigma, sigma_pert) to model LLM behavior. It swept beta1 to simulate different model accuracy levels, beta_gender as a fixed directional perturbation shift, and sigma as baseline noise. The perturbation arm already had separate noise (sigma_pert, defaulting to sigma).

Problems with v1:
- beta1 is an artificial proxy for model accuracy — the real logit distribution is richer (multimodal, heterogeneous confidence)
- beta_gender modeled a fixed directional shift, but real perturbations may affect different cases in different directions
- No connection to actual experimental data

The v2 simulation makes two structural changes: (1) real logits from MedPerturb experiments replace the synthetic beta0 + beta1 * z model, and (2) the directional beta_gender parameter is replaced by a variance-based sigma_pert that models perturbation sensitivity without assuming a consistent direction. The sigma sweep for baseline noise is retained.

---

## Data Generating Process

For each case i in a given (question, model) condition, with real logit z_i from the experiment:

```
epsilon_pert_i  ~ N(0, sigma_pert^2)    # perturbation-specific effect
epsilon_noise_i ~ N(0, sigma^2)          # general noise, perturbation arm
epsilon_noise_i'~ N(0, sigma^2)          # general noise, baseline arm (independent draw)

logit_pert_i = z_i + epsilon_pert_i + epsilon_noise_i
logit_base_i = z_i + epsilon_noise_i'

p_orig_i = sigmoid(z_i)                 # fixed from real data
p_pert_i = sigmoid(logit_pert_i)
p_base_i = sigmoid(logit_base_i)

y_orig_i = real binary answer            # fixed from real data
y_pert_i ~ Bernoulli(p_pert_i)
y_base_i ~ Bernoulli(p_base_i)
```

### Why logit space

Noise is additive in logit space (not probability space) because:
- **No boundary issues**: sigmoid maps any real value to (0, 1)
- **Confident predictions are naturally robust**: the sigmoid is flat in the tails, so a given noise magnitude has less effect on confident predictions — this is realistic
- **Clean component separation**: Gaussian components sum cleanly, so epsilon_pert and epsilon_noise are independent and decomposable. This would not hold for e.g. Beta distribution noise.

### Why sweep variance (not mean)

epsilon_pert is centered at zero with swept variance, rather than a fixed directional shift. This models perturbation effects as additional noise — the model becomes less stable when perturbed, but not necessarily in a consistent direction. This aligns with what MedPerturb tests: whether a perturbation causes more divergence than a calibrated baseline, regardless of the direction of change. All 5 metrics detect sensitivity/instability, not directional bias.

A variance-only perturbation (mean zero) still creates a detectable mean difference in JSD/KL: JSD(p, q) is non-negative and equals zero only when p = q. More variance in the perturbation arm means larger deviations from p_orig on average, so E[JSD(p_orig, p_pert)] > E[JSD(p_orig, p_base)] when the perturbation arm has more total noise. The paired t-test detects this mean difference.

### Null and alternative hypotheses

- **H0**: sigma_pert = 0. Both arms have only general noise. Symmetric.
- **H1**: sigma_pert > 0. Perturbation arm has additional noise beyond baseline.

### Where z_i comes from

The real logits are derived from MedPerturb experiment results stored in:
- `results/main_evaluation_llama_3.1_8b_instruct.json`
- `results/main_evaluation_llama_3.1_70b_instruct.json`

Each file is a list of dicts. For a given (question, model) condition, z_i comes from the `original_{QUESTION}` key's `logit_probs` field (a single float P(Yes)). The evaluation code (`evaluate_models.py`) extracts logits for "Yes" and "No" tokens, then applies 2-token softmax to get P(Yes). The conversion is z_i = log(P(Yes) / (1 - P(Yes))). For binary 2-token softmax, this equals l_Yes - l_No regardless of whether computed before or after softmax.

The binary answer y_orig_i is the majority vote across 3 seeds from `original_{QUESTION}.binary_answers` (matching the real experiment protocol).

### Handling extreme probabilities

Some cases have P(Yes) at or near 0 or 1, which produces extreme or infinite logits. The 70B model has P(Yes) = 1.0 in 1 MANAGE case, 17 VISIT cases, and 49 RESOURCE cases (67 total across all questions). The 8B model has no exact 0 or 1 values.

- **Clamp** P(Yes) to [epsilon, 1 - epsilon] before taking the logit, where epsilon = 1e-6. This maps P(Yes) = 1.0 to z_i ≈ 13.8.
- Cases with extreme z_i contribute almost nothing to metric sensitivity because the sigmoid is nearly flat — noise barely changes the probability. The effective sample size (cases with |z_i| < 3) varies substantially across conditions:

| Condition | Borderline (|z_i| < 3) | Total |
|-----------|----------------------|-------|
| 8B MANAGE | 72 | 100 |
| 8B VISIT | 96 | 100 |
| 8B RESOURCE | 50 | 100 |
| 70B MANAGE | 26 | 100 |
| 70B VISIT | 39 | 100 |
| 70B RESOURCE | 17 | 100 |

The 70B conditions will have substantially less power than 8B at the same sigma_pert due to fewer borderline cases.

Note: y_orig is fixed from real data while y_pert/y_base are stochastic Bernoulli draws. This is intentional — it isolates the comparison between the two simulated arms without adding a third source of sampling variance. The real experiment has three stochastic responses, so the simulation's variance structure differs slightly, but this does not affect the relative power comparison between metrics.

---

## Experimental Conditions

MedPerturb has 4 perturbations × 3 questions × 2 models:

**Perturbations:** Gender-Swap, Gender-Remove, Uncertain, Colorful
**Questions:** MANAGE, VISIT, RESOURCE
**Models:** Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct

The 4 perturbations are covered by the sigma_pert sweep — different perturbation types correspond to different points along the sigma_pert axis. So the simulation runs per (question, model) pair:

**6 simulations:** 3 questions × 2 models

Each simulation uses the real logits and binary answers from that condition's original prompts. The sample size is n = 100 for all 6 conditions (both models have 100 cases).

---

## Parameter Grid

**Swept parameters:**
- `sigma_pert`: perturbation effect strength. Covers different perturbation types/magnitudes.
- `sigma`: baseline noise level. Covers different baseline types (calibrated paraphrase vs fixed sentence).

**Fixed per condition:**
- `z_i`: real logits from MedPerturb experiment results
- `n`: real sample size for that (question, model) pair
- `y_orig_i`: real binary answers (majority vote)

**Sweep ranges (to be calibrated):**
- `sigma_pert`: 0 to 3.0, step 0.1. Generated as `[i / 10 for i in range(31)]` to avoid floating-point artifacts from `np.arange` (e.g., `0.30000000000000004`), which would break checkpoint resumption.
- `sigma`: {0.0, 0.25, 0.5, 1.0}

**Monte Carlo parameters:**
- `n_simulations`: 1000 per (sigma_pert, sigma) combo
- `n_bootstrap`: 1000 for per-population metric bootstrap tests

Total combos per condition: 31 × 4 = 124. With 6 conditions: 744 total combos.

---

## Metrics and Statistical Tests

Same 5 metrics and tests as v1.

### Per-population metrics (binary vectors y_orig, y_pert, y_base)

- **MI** (mutual information): bootstrap test
- **Phi** (phi coefficient): bootstrap test
- **Flip Rate**: bootstrap test

### Per-sample metrics (probability vectors p_orig, p_pert, p_base)

- **JSD**: paired t-test on JSD(p_orig_i, p_pert_i) vs JSD(p_orig_i, p_base_i)
- **KL**: paired t-test on KL(p_orig_i, p_pert_i) vs KL(p_orig_i, p_base_i)

### Difference from v1

In v1, p_orig was generated synthetically. In v2, p_orig = sigmoid(z_i) is fixed from real data. Similarly, y_orig is the real binary answer. Only p_pert, p_base, y_pert, y_base are simulated.

---

## Output and Presentation

### Per condition (6 total)

6 figures (one per condition), each with 4 subplots (one per sigma value). Within each subplot:
- X-axis: sigma_pert (perturbation effect strength)
- Y-axis: detection rate (0 to 1)
- 5 curves: one per metric, directly comparable
- Reference lines: alpha = 0.05, 80% power threshold

### What to verify

1. **Null calibration**: at sigma_pert = 0, detection rate ≈ 0.05 for all metrics and sigma > 0
2. **Power ordering**: per-sample metrics (JSD, KL) should have substantially more power than per-population metrics. Under the variance-sweep model, per-population metrics (especially flip rate) may have near-zero power because Bernoulli discretization destroys most of the signal.
3. **Noise degradation**: higher sigma should reduce power
4. **sigma=0 degeneracy**: at sigma = 0, the baseline arm has zero noise (p_base = p_orig exactly). Per-sample metrics see a step function from 0% to 100% detection at any sigma_pert > 0. Per-population metrics still have gradual power curves because Bernoulli sampling adds natural noise. The sigma=0 column serves as a boundary check but is not scientifically informative for per-sample metrics.

### Test sidedness

The existing metric implementations use two-sided tests (scipy `ttest_rel` for JSD/KL, two-sided bootstrap for MI/Phi/Flip Rate). The variance-sweep model guarantees the perturbation arm has equal or greater divergence than baseline (one-directional effect), so one-sided tests would have more power. The simulation uses two-sided tests to match the real analysis pipeline. This means the power estimates are conservative.

### For the paper

Present 1-2 representative conditions in main text, rest in appendix. Select representative conditions before seeing results (e.g., one high-accuracy condition and one low-accuracy condition) to avoid cherry-picking.

---

## Implementation

### Data loading

Load real logits and binary answers from:
- `results/main_evaluation_llama_3.1_8b_instruct.json`
- `results/main_evaluation_llama_3.1_70b_instruct.json`

For each (question, model) condition, extract:
- `logit_probs` from `original_{QUESTION}` → clamp to [1e-6, 1-1e-6] → convert via `log(p / (1 - p))` → z_i array
- `binary_answers` from `original_{QUESTION}` → majority vote → y_orig array

### Parallelization

Same approach as v1: `multiprocessing.Pool` across (sigma_pert, sigma) combos. Each combo is independent. Checkpoint CSV every 10 combos for resumption.

### Checkpoint format

One CSV per condition. Columns: `metric, sigma_pert, sigma, condition, detection_rate, mean_p_value`. The `condition` column encodes the (question, model) pair as `{QUESTION}_{model_short_name}` (e.g., `MANAGE_8b`, `VISIT_70b`).

### Deterministic seeding

Same as v1: SHA256-based per-combo seed. Hash key format: `"{global_seed}:{sigma_pert:.1f}:{sigma:.2f}:{condition}"` using formatted float strings to avoid floating-point representation mismatches between generation and checkpoint resumption (e.g., `"42:0.50:0.25:MANAGE_8b"`). Independent of execution order.

### SLURM

CPU-only job on 177huntington partition, 32 cores, 32G memory. One job iterating over all 6 conditions sequentially (each condition's combos parallelized across 32 workers).

### Output directory

Results saved to `results/simulation_v2/`. Per-condition files:
- CSV: `simulation_v2_{condition}.csv` (e.g., `simulation_v2_MANAGE_8b.csv`)
- Figures: `power_curves_{condition}.png` (e.g., `power_curves_MANAGE_8b.png`)

---

## Files

### Modified
| File | Change |
|------|--------|
| `code/simulation.py` | Rewrite DGP to use real logits, new parameter grid |
| `slurm/run_simulation.sbatch` | Update CLI args for new parameters |

### New (if needed)
| File | Purpose |
|------|---------|
| `code/load_experiment_logits.py` | Extract z_i and y_orig from experiment results (if complex enough to warrant separation) |
