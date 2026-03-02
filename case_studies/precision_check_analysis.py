# ABOUTME: Bootstrap MI analysis for precision check (age swap negative control)
# ABOUTME: Validates that age swap does NOT affect gender detection, confirming measurement precision

import numpy as np
import pandas as pd
import json
import argparse
import sys

sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/case_studies')
from baseline_analysis import bootstrap_mi_test


def majority_vote(responses):
    """Return majority vote from list of 0/1 responses."""
    return 1 if sum(responses) > len(responses) / 2 else 0


def run_precision_check_analysis(eval_results, n_bootstrap=1000):
    """
    Run MI analysis on precision check evaluation results.

    Args:
        eval_results: List of dicts from precision_check_evaluate.py output
        n_bootstrap: Number of bootstrap iterations

    Returns:
        dict with mi_perturbation, mi_baseline, observed_diff, ci_low, ci_high, p_value, n_cases
    """
    orig_votes = []
    age_swap_votes = []
    baseline_votes = []

    for r in eval_results:
        orig_votes.append(majority_vote(r['original_GENDER']['binary_answers']))
        age_swap_votes.append(majority_vote(r['age_swap_GENDER']['binary_answers']))
        baseline_votes.append(majority_vote(r['age_swap_baseline_GENDER']['binary_answers']))

    orig = pd.Series(orig_votes)
    age_swap = pd.Series(age_swap_votes)
    baseline = pd.Series(baseline_votes)

    result = bootstrap_mi_test(orig, age_swap, baseline, n_bootstrap=n_bootstrap)
    result['n_cases'] = len(eval_results)

    return result


def main():
    parser = argparse.ArgumentParser(description="Precision check MI analysis (age swap negative control)")
    parser.add_argument('--evaluation', type=str, required=True,
                        help='Path to precision check evaluation JSON')
    parser.add_argument('--output', type=str, default='results/precision_check_analysis.xlsx',
                        help='Output Excel path')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations')

    args = parser.parse_args()

    with open(args.evaluation, 'r') as f:
        eval_results = json.load(f)
    print(f"Loaded {len(eval_results)} evaluation results")

    result = run_precision_check_analysis(eval_results, n_bootstrap=args.n_bootstrap)

    results_df = pd.DataFrame([result])
    results_df.to_excel(args.output, index=False)

    print("\n" + "=" * 60)
    print("Precision Check Results (Age Swap Negative Control)")
    print("=" * 60)
    print(f"N cases:              {result['n_cases']}")
    print(f"MI(orig, age_swap):   {result['mi_perturbation']:.4f}")
    print(f"MI(orig, base):       {result['mi_baseline']:.4f}")
    print(f"Difference:           {result['observed_diff']:+.4f}")
    print(f"95% CI:               [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]")
    print(f"p-value:              {result['p_value']:.4f}")

    sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
    print(f"Significance:         {sig}")

    if result['p_value'] > 0.05:
        print("\nPRECISION CHECK PASSED: Age swap does not significantly affect gender detection (p > 0.05)")
    else:
        print(f"\nWARNING: Age swap significantly affects gender detection (p = {result['p_value']:.4f})")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
