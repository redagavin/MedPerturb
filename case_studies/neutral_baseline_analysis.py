# ABOUTME: Bootstrap MI analysis comparing perturbation effects vs neutral sentence baseline
# ABOUTME: Tests whether perturbations have specific effects beyond adding irrelevant text

import numpy as np
import pandas as pd
import argparse


def calculate_mi(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate mutual information between two binary arrays.

    Uses pd.crosstab for joint distribution, then MI = sum of
    p(x,y) * log2(p(x,y) / (p(x)*p(y))).

    Args:
        x: First array (binary 0/1)
        y: Second array (binary 0/1)

    Returns:
        float: Mutual information in bits
    """
    joint = pd.crosstab(x, y, normalize=True)
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    mi = 0
    for i in joint.index:
        for j in joint.columns:
            if joint.loc[i, j] > 0:
                mi += joint.loc[i, j] * np.log2(joint.loc[i, j] / (p_x[i] * p_y[j]))
    return mi


def bootstrap_mi_test(
    orig: pd.Series,
    pert: pd.Series,
    base: pd.Series,
    n_bootstrap: int = 1000
) -> dict:
    """
    Bootstrap hypothesis test comparing MI(orig,pert) vs MI(orig,base).

    Args:
        orig: Original responses
        pert: Perturbation responses
        base: Baseline (neutral sentence) responses
        n_bootstrap: Number of bootstrap iterations

    Returns:
        dict: {
            'mi_perturbation': float,
            'mi_baseline': float,
            'observed_diff': float,
            'ci_low': float,
            'ci_high': float,
            'p_value': float
        }
    """
    n = len(orig)
    diffs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        mi_pert = calculate_mi(
            orig.iloc[indices].reset_index(drop=True),
            pert.iloc[indices].reset_index(drop=True)
        )
        mi_base = calculate_mi(
            orig.iloc[indices].reset_index(drop=True),
            base.iloc[indices].reset_index(drop=True)
        )
        diffs.append(mi_pert - mi_base)

    observed_diff = calculate_mi(orig, pert) - calculate_mi(orig, base)

    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)

    if observed_diff >= 0:
        p_value = np.mean(np.array(diffs) <= 0) * 2
    else:
        p_value = np.mean(np.array(diffs) >= 0) * 2
    p_value = min(p_value, 1.0)

    return {
        'mi_perturbation': calculate_mi(orig, pert),
        'mi_baseline': calculate_mi(orig, base),
        'observed_diff': observed_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_value
    }


def run_analysis(dataset_path: str, output_path: str):
    """
    Run full bootstrap MI analysis comparing perturbations vs neutral baseline.

    All perturbation types are compared against the single neutral baseline
    (dataset_id=6) to test whether perturbations have specific effects beyond
    adding irrelevant text.

    Args:
        dataset_path: Path to data_with_neutral_baseline.csv
        output_path: Path for output Excel file
    """
    df = pd.read_csv(dataset_path)
    df = df[df['dataset'] != 'conversational']

    perturbation_types = {
        2: 'gender_swap',
        3: 'gender_remove',
        4: 'uncertain',
        5: 'colorful'
    }

    models = ['LLAMA3', 'LLAMA3-70']
    tasks = ['MANAGE', 'VISIT', 'RESOURCE']

    results = []

    for pert_id, pert_name in perturbation_types.items():
        for model in models:
            for task in tasks:
                col = f'{model}_{task}'

                originals = df[df['dataset_id'] == 1].set_index(['dataset', 'context_id'])
                perturbations = df[df['dataset_id'] == pert_id].set_index(['dataset', 'context_id'])
                baselines = df[df['dataset_id'] == 6].set_index(['dataset', 'context_id'])

                common_idx = originals.index.intersection(perturbations.index).intersection(baselines.index)

                if len(common_idx) == 0:
                    print(f"Warning: No common cases for {pert_name}, {model}, {task}")
                    continue

                orig_vals = originals.loc[common_idx, col].reset_index(drop=True)
                pert_vals = perturbations.loc[common_idx, col].reset_index(drop=True)
                base_vals = baselines.loc[common_idx, col].reset_index(drop=True)

                if orig_vals.isna().any() or pert_vals.isna().any() or base_vals.isna().any():
                    print(f"Warning: NaN values for {pert_name}, {model}, {task}")
                    continue

                test_result = bootstrap_mi_test(orig_vals, pert_vals, base_vals)

                results.append({
                    'perturbation_type': pert_name,
                    'model': model,
                    'task': task,
                    'n_cases': len(common_idx),
                    **test_result
                })

    results_df = pd.DataFrame(results)
    results_df.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for _, row in results_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['perturbation_type']:15} {row['model']:10} {row['task']:10} "
              f"diff={row['observed_diff']:+.4f} p={row['p_value']:.4f} {sig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap MI analysis: perturbation vs neutral sentence baseline"
    )
    parser.add_argument('--dataset', type=str, default='data_with_neutral_baseline.csv',
                        help='Path to dataset with neutral baseline')
    parser.add_argument('--output', type=str, default='results/neutral_baseline_analysis.xlsx',
                        help='Output Excel path')

    args = parser.parse_args()

    run_analysis(args.dataset, args.output)
