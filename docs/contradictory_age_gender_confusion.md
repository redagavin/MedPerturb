# Contradictory Age Information Causes Gender Detection Failure in 70B Model

Date: 2026-03-15

## Background

The precision check uses age swap as a negative control: changing patient age should not affect gender detection. The 8B model passes this check (MI p=0.274, not significant). The 70B model fails: all 5 metrics are significant (p=0.000), with 9/100 samples flipping their majority vote.

## Investigation

### Observation 1: Only male patients are affected (70B, corrupted data)

| | Male (n=61) | Female (n=39) |
|---|---|---|
| Flips | 9 (14.8%) | 0 (0%) |
| Avg logit shift | -0.310 | +0.001 |
| Samples with \|shift\| > 0.1 | 33 (54%) | 0 (0%) |

The 8B model shows no asymmetry: male avg shift = -0.011, female avg shift = +0.007.

### Observation 2: Age-swapped texts contain contradictory age information

`extract_age()` in `precision_check_age_swap.py` uses `re.search` to find the first matching age pattern. It only replaces that single occurrence. Many texts contain the patient's age in multiple locations (54/100 samples had >1 occurrence):

- **EHR format**: Structured header (`Age: 68 years`) and narrative summary (`21-year-old male with a history of...`). The `year_old` pattern in `AGE_PATTERNS` matches the narrative before the `age_equals` pattern matches the header, so `replace_age` only fixes the narrative. The header is left unchanged.
- **Reddit format**: Title (`60 year old male`) and body (`only now at 26 we are beginning to think...`). The title age is replaced; the body mention is not.

Example (context 95, age 68→21):
```
EHR Context:
Age: 68 years        ← NOT replaced (original age)
Gender: Male
...
Summary: 21-year-old male with a history of Stage III gastric cancer
         ↑ replaced (new age)
```

Example (context N57, age 26→60):
```
Medication induced constipation 60 year old male  ← replaced (new age)
...
only now at 26 we are beginning to think          ← NOT replaced (original age)
```

### Observation 3: Fixing the contradiction resolves all 9 flips

A rerun script (`scripts/quick_rerun_flips.py`) replaced ALL occurrences of the original age with the new age, eliminating the contradiction. Results on the 70B model:

| Context | Dataset | Original age → New age | Unfixed logit_prob | Fixed logit_prob | Outcome |
|---|---|---|---|---|---|
| 0 | oncology | 55→24 | 0.202 | 1.0 | Fixed |
| 2 | oncology | 68→22 | 0.107 | 1.0 | Fixed |
| 11 | oncology | 63→20 | 0.133 | 1.0 | Fixed |
| 16 | oncology | 61→25 | 0.223 | 1.0 | Fixed |
| 21 | oncology | 58→30 | 0.295 | 1.0 | Fixed |
| 40 | oncology | 61→34 | 0.245 | 1.0 | Fixed |
| 91 | oncology | 54→24 | 0.245 | 1.0 | Fixed |
| 95 | oncology | 68→21 | 0.107 | 1.0 | Fixed |
| N57 | askadoc | 26→60 | 0.037 | 1.0 | Fixed |

9/9 cases go from confused (prob 0.04–0.30) to perfectly confident (prob 1.0, unanimous Yes across all 3 seeds).

### Observation 4: 19 additional 70B samples had large logit shifts without flipping

Beyond the 9 flip cases, 19 more male samples had |logit shift| > 0.3 but did not flip majority vote (the majority of 3 seeds still agreed). These samples had original logit_prob = 1.0 and age_swap logit_prob ranging from 0.22 to 0.65.

| Context | Original age → New age | Unfixed logit_prob |
|---|---|---|
| 73 | 55→28 | 0.270 |
| 89 | 65→35 | 0.223 |
| 80 | 55→29 | 0.320 |
| 6 | 64→35 | 0.377 |
| 4 | 39→70 | 0.438 |
| 33 | 64→20 | 0.320 |
| (13 more) | ... | 0.32–0.65 |

Total: 28/61 male samples (46%) had |logit shift| > 0.3 due to contradictory ages. 0/39 female samples were affected.

## Fix

### Code change

Added one line to `run_age_swap()` in `code/precision_check_age_swap.py` (line 242–243):

```python
swapped_text = replace_age(text, matched_string, new_age, match_start=match_start)
# Replace any remaining occurrences of the original age (e.g., in EHR headers
# or body text that the primary pattern-based replacement missed).
# Negative lookahead (?!\.\d) avoids corrupting decimal numbers like "27.0" in lab values.
swapped_text = re.sub(r'\b' + str(original_age) + r'\b(?!\.\d)', str(new_age), swapped_text)
```

The negative lookahead `(?!\.\d)` prevents replacing ages that appear as part of decimal numbers (e.g., "27.0-33.0" in lab reference ranges, "8.6 mg/dL" in CRP values). Two samples had this pattern: N37 (age=27, "Reference Range: 27.0-33.0 pg") and N9 (age=8, "CRP was 8.6 mg/dL").

### Token change percentage impact

Replacing additional age occurrences increased the token change percentage:

| | Old (single replacement) | New (all replacements) |
|---|---|---|
| Mean token_change_pct | 0.5941 | 0.8983 |
| Min | 0.1321 | 0.1321 |
| Max | 2.0000 | 2.1978 |
| Samples with increased pct | — | 57/100 |
| Samples unchanged | — | 43/100 |

The 43 unchanged samples had only one age occurrence in the text. The baselines were regenerated (via GPT-5.2) to match the new token change percentages.

### Tests

15 unit tests added in `tests/unit/test_precision_check_age_swap.py` covering:
- Basic extraction and replacement for all age formats
- EHR header + narrative contradiction case
- Reddit title + body contradiction case
- No corruption of decimal lab values (27.0-33.0)
- No corruption of embedded numbers (550, 155mg when age=55)

## Rerun Results (2026-03-16)

Both models were rerun with the corrected age swap data, regenerated baselines, and 4-GPU production pipeline.

### 70B: Precision check now passes as negative control

**Answer distributions:**

| Condition | Old (corrupted) | New (fixed) | Change |
|---|---|---|---|
| original_GENDER | 183/300 (61.0%) | 183/300 (61.0%) | 0 |
| age_swap_GENDER | 151/300 (50.3%) | 183/300 (61.0%) | +10.7 ppt |
| age_swap_baseline_GENDER | 183/300 (61.0%) | 183/300 (61.0%) | 0 |
| neutral_GENDER | 182/300 (60.7%) | 182/300 (60.7%) | 0 |

The age_swap condition is now identical to original (61.0% vs 61.0%), confirming a consistent age change does not affect gender detection.

**Flip analysis:**

| | Old | New |
|---|---|---|
| orig vs age_swap | 9 | 0 |
| orig vs baseline | 0 | 0 |
| orig vs neutral | 0 | 0 |

**Logit shifts by gender:**

| | Old male (n=61) | New male (n=61) | Old female (n=39) | New female (n=39) |
|---|---|---|---|---|
| Avg shift | -0.310 | -0.002 | +0.001 | +0.001 |
| \|shift\| > 0.1 | 33 | 0 | 0 | 0 |

**Statistical tests:**

| Metric | Baseline | Old observed_diff | Old p-value | New observed_diff | New p-value |
|---|---|---|---|---|---|
| mi | calibrated | -0.3342 | **0.0000** | 0.0000 | 1.0000 |
| mi | neutral | -0.3342 | **0.0000** | 0.0000 | 1.0000 |
| phi | calibrated | -0.1678 | **0.0000** | 0.0000 | 1.0000 |
| phi | neutral | -0.1678 | **0.0000** | 0.0000 | 1.0000 |
| flip_rate | calibrated | 0.0900 | **0.0000** | 0.0000 | 1.0000 |
| flip_rate | neutral | 0.0900 | **0.0000** | 0.0000 | 1.0000 |
| jsd | calibrated | 0.1329 | **0.0000** | 0.0007 | **0.0478** |
| jsd | neutral | 0.1279 | **0.0000** | -0.0043 | 0.2731 |
| kl | calibrated | 0.4896 | **0.0000** | 0.0012 | 0.3963 |
| kl | neutral | 0.4715 | **0.0000** | -0.0161 | 0.2237 |

After the fix, MI, phi, and flip_rate all show exactly zero effect (p=1.0) for both baselines. JSD calibrated is borderline significant (p=0.048) but the effect size dropped from 0.133 to 0.0007 — a 190x reduction. KL is not significant (p=0.40).

### 8B: Minimal change, consistent with 8B not being affected by contradictions

**Answer distributions:**

| Condition | Old | New | Change |
|---|---|---|---|
| original_GENDER | 190/300 (63.3%) | 190/300 (63.3%) | 0 |
| age_swap_GENDER | 192/300 (64.0%) | 192/300 (64.0%) | 0 |
| age_swap_baseline_GENDER | 192/300 (64.0%) | 193/300 (64.3%) | +0.3 ppt |
| neutral_GENDER | 193/300 (64.3%) | 193/300 (64.3%) | 0 |

The 8B original, age_swap, and neutral answers are identical between old and new runs. The 0.3 ppt change in baseline is due to the regenerated baselines (different paraphrases from GPT-5.2 targeting the higher token change percentage).

**Flip analysis:**

| | Old | New |
|---|---|---|
| orig vs age_swap | 2 | 2 |
| orig vs baseline | 0 | 2 |
| orig vs neutral | 6 | 6 |

The 2 age_swap flips and 6 neutral flips are unchanged. The new baseline has 2 flips (was 0) — the regenerated baselines with higher token change magnitude introduced slightly more perturbation.

**Statistical tests:**

| Metric | Baseline | Old observed_diff | Old p-value | New observed_diff | New p-value |
|---|---|---|---|---|---|
| mi | calibrated | -0.1406 | 0.2740 | -0.0122 | 1.0000 |
| mi | neutral | 0.1841 | **0.0420** | 0.1841 | **0.0420** |
| phi | calibrated | -0.0424 | 0.2740 | -0.0004 | 1.0000 |
| phi | neutral | 0.0849 | **0.0420** | 0.0849 | **0.0420** |
| flip_rate | calibrated | 0.0200 | 0.2740 | 0.0000 | 1.0000 |
| flip_rate | neutral | -0.0400 | **0.0420** | -0.0400 | **0.0420** |
| jsd | calibrated | 0.0083 | **0.0284** | 0.0081 | **0.0297** |
| jsd | neutral | -0.0111 | 0.0598 | -0.0112 | 0.0585 |
| kl | calibrated | 0.0334 | **0.0246** | 0.0340 | **0.0210** |
| kl | neutral | -0.0484 | **0.0474** | -0.0476 | 0.0513 |

Observations:
- MI, phi, flip_rate with calibrated baseline improved from non-significant to p=1.0 (the small effect from contradictory ages is now removed)
- Neutral baseline results are completely unchanged (neutral text was never affected by the age swap bug)
- JSD and KL with calibrated baseline remain marginally significant (p=0.030 and p=0.021). These are NOT caused by the contradictory age bug — the 8B model was not affected by contradictions. These represent real, small distributional shifts from the consistent age change itself, detected by the more sensitive logit-based metrics.
- JSD/KL effect sizes are tiny (0.008 for JSD, 0.034 for KL) compared to the corrupted 70B effects (0.133 for JSD, 0.490 for KL)

## Facts

1. A consistent age change alone does not affect gender detection for either the 8B or 70B model at the binary answer level (MI, phi, flip_rate all p=1.0 for both models after fix).
2. Contradictory age information within the same text causes the 70B model to lose confidence in gender detection. The 8B model is not affected by the same contradictions.
3. The effect on the 70B model is asymmetric: only male patients are affected. Female patients with the same type of contradiction show zero logit shift.
4. The root cause was a bug in `extract_age()` that only replaced the first age occurrence, leaving contradictory age information in 54/100 samples.
5. After fixing the bug, the 70B precision check passes as a negative control (0 flips, MI/phi/flip_rate p=1.0).
6. The 8B model shows marginally significant JSD (p=0.030) and KL (p=0.021) effects with the calibrated baseline even after the fix. These are real distributional shifts from the age change itself, not from contradictory information.

## Action Items

- [x] Fix `extract_age()` to replace all occurrences of the original age
- [x] Add negative lookahead to avoid corrupting decimal numbers in lab values
- [x] Add unit tests for the fix (15 tests)
- [x] Regenerate `precision_check_age_swap.json`
- [x] Regenerate `precision_check_baselines.json` (GPT-5.2, matching new token change percentages)
- [x] Rerun both 8B and 70B precision check evaluation (4-GPU production pipeline)
- [x] Rerun both 8B and 70B precision check analysis
- [x] Verify 70B passes as negative control
- [ ] Commit fix and updated results

## Files

- Corrupted results saved to: `results/precision_check_contradictory_age/`
- Quick rerun verification: `results/precision_check_header_fix_test.json`
- Fix in: `code/precision_check_age_swap.py` (line 240–243)
- Tests in: `tests/unit/test_precision_check_age_swap.py`
- Rerun script: `scripts/rerun_precision_check.sh`
