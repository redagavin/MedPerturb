# ABOUTME: Monte Carlo power simulation for all 5 MedPerturb metrics.
# ABOUTME: Estimates false positive rate and statistical power via the logistic generative model.

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid

from metrics import mi, phi, flip_rate, jsd, kl


ALL_METRICS = ["mi", "phi", "flip_rate", "jsd", "kl"]


def generate_responses(
    n_cases: int,
    beta0: float,
    beta1: float,
    beta_gender: float,
    rng: np.random.Generator,
    sigma: float = 0.0,
) -> dict:
    """Generate synthetic responses from the logistic generative model.

    For each case i:
        true_label_i ~ Bernoulli(0.5)
        z_i = 2 * true_label_i - 1  (maps {0,1} to {-1,+1})
        p_orig_i = sigmoid(beta0 + beta1 * z_i)
        p_pert_i = sigmoid(beta0 + beta1 * z_i + beta_gender + epsilon_pert_i)
        p_base_i = sigmoid(beta0 + beta1 * z_i + epsilon_base_i)
        y_orig ~ Bernoulli(p_orig_i), y_pert ~ Bernoulli(p_pert_i), etc.

    Returns dict with both binary vectors (for per-population metrics)
    and probability vectors (for per-sample metrics).
    """
    true_labels = rng.binomial(1, 0.5, size=n_cases)
    z = 2 * true_labels - 1

    logit_orig = beta0 + beta1 * z

    # Both arms get independent noise so they are symmetric under H0
    epsilon_pert = rng.normal(0, sigma, size=n_cases) if sigma > 0 else 0.0
    epsilon_base = rng.normal(0, sigma, size=n_cases) if sigma > 0 else 0.0
    logit_pert = logit_orig + beta_gender + epsilon_pert
    logit_base = logit_orig + epsilon_base

    p_orig = sigmoid(logit_orig)
    p_pert = sigmoid(logit_pert)
    p_base = sigmoid(logit_base)

    orig = rng.binomial(1, p_orig)
    pert = rng.binomial(1, p_pert)
    base = rng.binomial(1, p_base)

    return {
        "orig": orig,
        "pert": pert,
        "base": base,
        "p_orig": p_orig,
        "p_pert": p_pert,
        "p_base": p_base,
    }


def run_single_simulation(
    data: dict,
    n_bootstrap: int = 1000,
    rng: np.random.Generator = None,
) -> dict:
    """Run all 5 metric tests on one simulated dataset.

    Per-population metrics (MI, Phi, Flip Rate) use bootstrap tests on binary vectors.
    Per-sample metrics (JSD, KL) use paired t-tests on probability vectors.

    Returns dict mapping metric name to p-value.
    """
    if rng is None:
        rng = np.random.default_rng()

    orig_bin = np.asarray(data["orig"])
    pert_bin = np.asarray(data["pert"])
    base_bin = np.asarray(data["base"])

    p_orig = np.asarray(data["p_orig"])
    p_pert = np.asarray(data["p_pert"])
    p_base = np.asarray(data["p_base"])

    boot_seed = int(rng.integers(0, 2**31))

    # Per-population metrics: bootstrap tests on binary vectors
    mi_result = mi.bootstrap_test(orig_bin, pert_bin, base_bin,
                                  n_bootstrap=n_bootstrap, seed=boot_seed)
    phi_result = phi.bootstrap_test(orig_bin, pert_bin, base_bin,
                                    n_bootstrap=n_bootstrap, seed=boot_seed)
    flip_result = flip_rate.bootstrap_test(orig_bin, pert_bin, base_bin,
                                           n_bootstrap=n_bootstrap, seed=boot_seed)

    # Per-sample metrics: paired t-tests on probability vectors
    jsd_result = jsd.paired_ttest(p_orig, p_pert, p_base)
    kl_result = kl.paired_ttest(p_orig, p_pert, p_base)

    return {
        "mi": mi_result["p_value"],
        "phi": phi_result["p_value"],
        "flip_rate": flip_result["p_value"],
        "jsd": jsd_result["p_value"],
        "kl": kl_result["p_value"],
    }


def run_power_analysis(
    beta1_values: list[float],
    beta_gender_values: list[float],
    n_simulations: int,
    n_cases: int = 100,
    n_bootstrap: int = 1000,
    beta0: float = 0.0,
    seed: int = 42,
    sigma_values: list[float] = None,
) -> pd.DataFrame:
    """Run Monte Carlo simulation across a parameter grid for all 5 metrics.

    For each (beta1, sigma, beta_gender) combination, runs n_simulations
    synthetic experiments and records how often each metric test rejects
    at p < 0.05.

    Returns DataFrame with columns: metric, beta1, sigma, beta_gender,
    detection_rate, mean_p_value.
    """
    if sigma_values is None:
        sigma_values = [0.0]

    results = []
    base_rng = np.random.default_rng(seed)

    for beta1 in beta1_values:
        for sigma in sigma_values:
            for beta_gender in beta_gender_values:
                p_values_by_metric = {m: [] for m in ALL_METRICS}

                for _ in range(n_simulations):
                    sim_seed = base_rng.integers(0, 2**31)
                    sim_rng = np.random.default_rng(sim_seed)

                    data = generate_responses(
                        n_cases, beta0, beta1, beta_gender, sim_rng,
                        sigma=sigma,
                    )
                    boot_rng = np.random.default_rng(sim_rng.integers(0, 2**31))
                    pvals = run_single_simulation(
                        data, n_bootstrap=n_bootstrap, rng=boot_rng,
                    )
                    for m in ALL_METRICS:
                        p_values_by_metric[m].append(pvals[m])

                for m in ALL_METRICS:
                    pv = np.array(p_values_by_metric[m])
                    results.append({
                        "metric": m,
                        "beta1": beta1,
                        "sigma": sigma,
                        "beta_gender": beta_gender,
                        "detection_rate": float(np.mean(pv < 0.05)),
                        "mean_p_value": float(np.mean(pv)),
                    })

    return pd.DataFrame(results)


def generate_power_curves(results: pd.DataFrame, output_dir: str) -> None:
    """Generate one power curve figure per (metric, sigma) combination.

    Plots detection rate vs beta_gender, with one line per beta1 value.
    """
    for metric in sorted(results["metric"].unique()):
        metric_data = results[results["metric"] == metric]
        for sigma in sorted(metric_data["sigma"].unique()):
            subset = metric_data[metric_data["sigma"] == sigma]
            fig, ax = plt.subplots(figsize=(8, 5))

            for beta1 in sorted(subset["beta1"].unique()):
                b1_sub = subset[subset["beta1"] == beta1].sort_values("beta_gender")
                accuracy_pct = int(round(sigmoid(beta1) * 100))
                ax.plot(
                    b1_sub["beta_gender"], b1_sub["detection_rate"],
                    marker='o', markersize=4,
                    label=f'\u03b2\u2081={beta1} (~{accuracy_pct}% accuracy)',
                )

            ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1,
                        label='\u03b1 = 0.05')
            ax.axhline(y=0.80, color='lightgray', linestyle=':',
                        linewidth=1, label='80% power')
            ax.set_xlabel('\u03b2_gender (bias coefficient)')
            ax.set_ylabel('Detection rate')
            ax.set_title(f'Power: {metric} (\u03c3={sigma})')
            ax.legend()
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            filename = f'power_curve_{metric}_sigma_{sigma}.png'
            fig.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Monte Carlo power simulation for all 5 MedPerturb metrics"
    )
    parser.add_argument('--beta1-values', type=float, nargs='+',
                        default=[1.0, 2.0, 3.0],
                        help='Model accuracy coefficients')
    parser.add_argument('--beta-gender-max', type=float, default=4.0,
                        help='Maximum beta_gender to sweep')
    parser.add_argument('--beta-gender-step', type=float, default=0.1,
                        help='Step size for beta_gender sweep')
    parser.add_argument('--n-simulations', type=int, default=1000,
                        help='Number of simulation runs per parameter combo')
    parser.add_argument('--n-cases', type=int, default=100,
                        help='Number of cases per simulated experiment')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Bootstrap iterations per test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--sigma-values', type=float, nargs='+',
                        default=[0.0],
                        help='Baseline noise levels (sigma)')
    parser.add_argument('--output-dir', type=str,
                        default='results/simulation',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    beta_gender_values = list(np.arange(0, args.beta_gender_max + 1e-9,
                                        args.beta_gender_step))

    total_combos = (len(args.beta1_values) * len(beta_gender_values)
                    * len(args.sigma_values))
    print(f"Running power analysis:")
    print(f"  beta1 values: {args.beta1_values}")
    print(f"  sigma values: {args.sigma_values}")
    print(f"  beta_gender range: 0 to {args.beta_gender_max} "
          f"(step {args.beta_gender_step})")
    print(f"  {args.n_simulations} simulations per combination")
    print(f"  {args.n_cases} cases, {args.n_bootstrap} bootstrap iterations")
    print(f"  Total parameter combos: {total_combos}")
    print(f"  Metrics: {ALL_METRICS}")

    results = run_power_analysis(
        beta1_values=args.beta1_values,
        beta_gender_values=beta_gender_values,
        sigma_values=args.sigma_values,
        n_simulations=args.n_simulations,
        n_cases=args.n_cases,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    results_path = os.path.join(args.output_dir, 'simulation_results.csv')
    results.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    generate_power_curves(results, args.output_dir)
    print(f"Power curves saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
