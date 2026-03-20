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
    sigma_pert: float = None,
) -> dict:
    """Generate synthetic responses from the logistic generative model.

    For each case i:
        true_label_i ~ Bernoulli(0.5)
        z_i = 2 * true_label_i - 1  (maps {0,1} to {-1,+1})
        p_orig_i = sigmoid(beta0 + beta1 * z_i)
        p_pert_i = sigmoid(beta0 + beta1 * z_i + beta_gender + epsilon_pert_i)
        p_base_i = sigmoid(beta0 + beta1 * z_i + epsilon_base_i)
        y_orig ~ Bernoulli(p_orig_i), y_pert ~ Bernoulli(p_pert_i), etc.

    sigma controls baseline noise (paraphrase/insertion variation).
    sigma_pert controls perturbation noise (defaults to sigma if not specified).

    Returns dict with both binary vectors (for per-population metrics)
    and probability vectors (for per-sample metrics).
    """
    if sigma_pert is None:
        sigma_pert = sigma

    true_labels = rng.binomial(1, 0.5, size=n_cases)
    z = 2 * true_labels - 1

    logit_orig = beta0 + beta1 * z

    epsilon_pert = rng.normal(0, sigma_pert, size=n_cases) if sigma_pert > 0 else 0.0
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


def generate_responses_v2(
    z_i: np.ndarray,
    y_orig: np.ndarray,
    sigma_pert: float,
    sigma: float,
    rng: np.random.Generator,
) -> dict:
    """Generate simulated responses from real logits.

    For each case i with real logit z_i:
        epsilon_pert_i  ~ N(0, sigma_pert^2)
        epsilon_noise_i ~ N(0, sigma^2)       (perturbation arm)
        epsilon_noise_i'~ N(0, sigma^2)       (baseline arm, independent)

        logit_pert_i = z_i + epsilon_pert_i + epsilon_noise_i
        logit_base_i = z_i + epsilon_noise_i'

        p_pert = sigmoid(logit_pert), p_base = sigmoid(logit_base)
        y_pert ~ Bernoulli(p_pert), y_base ~ Bernoulli(p_base)

    Returns dict with binary vectors and probability vectors.
    """
    n = len(z_i)

    epsilon_pert = rng.normal(0, sigma_pert, size=n) if sigma_pert > 0 else 0.0
    epsilon_noise_pert = rng.normal(0, sigma, size=n) if sigma > 0 else 0.0
    epsilon_noise_base = rng.normal(0, sigma, size=n) if sigma > 0 else 0.0

    logit_pert = z_i + epsilon_pert + epsilon_noise_pert
    logit_base = z_i + epsilon_noise_base

    p_orig = sigmoid(z_i)
    p_pert = sigmoid(logit_pert)
    p_base = sigmoid(logit_base)

    pert_bin = rng.binomial(1, p_pert)
    base_bin = rng.binomial(1, p_base)

    return {
        "orig": y_orig.copy(),
        "pert": pert_bin,
        "base": base_bin,
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


def _combo_seed(global_seed, beta1, sigma, beta_gender):
    """Derive a deterministic seed from parameters, independent of execution order."""
    import hashlib
    key = f"{global_seed}:{beta1}:{sigma}:{beta_gender}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def _run_one_combo(args):
    """Run all simulations for one (beta1, sigma, beta_gender) combo.

    Designed as a top-level function for multiprocessing.Pool.
    """
    beta1, sigma, beta_gender, n_simulations, n_cases, n_bootstrap, beta0, global_seed, sigma_pert = args

    combo_seed = _combo_seed(global_seed, beta1, sigma, beta_gender)
    base_rng = np.random.default_rng(combo_seed)

    p_values_by_metric = {m: [] for m in ALL_METRICS}

    for _ in range(n_simulations):
        sim_seed = base_rng.integers(0, 2**31)
        sim_rng = np.random.default_rng(sim_seed)

        data = generate_responses(
            n_cases, beta0, beta1, beta_gender, sim_rng,
            sigma=sigma, sigma_pert=sigma_pert,
        )
        boot_rng = np.random.default_rng(sim_rng.integers(0, 2**31))
        pvals = run_single_simulation(
            data, n_bootstrap=n_bootstrap, rng=boot_rng,
        )
        for m in ALL_METRICS:
            p_values_by_metric[m].append(pvals[m])

    results = []
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
    return results


def _combo_seed_v2(global_seed, sigma_pert, sigma, condition):
    """Derive a deterministic seed from v2 parameters.

    Uses formatted float strings (:.1f for sigma_pert, :.2f for sigma) to ensure
    floating-point representation artifacts don't produce different seeds.
    """
    import hashlib
    key = f"{global_seed}:{sigma_pert:.1f}:{sigma:.2f}:{condition}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def _run_one_combo_v2(args):
    """Run all simulations for one (sigma_pert, sigma) combo using real logits.

    Designed as a top-level function for multiprocessing.Pool.
    """
    sigma_pert, sigma, condition, n_simulations, n_bootstrap, global_seed, z_i, y_orig = args

    combo_seed = _combo_seed_v2(global_seed, sigma_pert, sigma, condition)
    base_rng = np.random.default_rng(combo_seed)

    p_values_by_metric = {m: [] for m in ALL_METRICS}

    for _ in range(n_simulations):
        sim_seed = base_rng.integers(0, 2**31)
        sim_rng = np.random.default_rng(sim_seed)

        data = generate_responses_v2(z_i, y_orig, sigma_pert, sigma, sim_rng)
        boot_rng = np.random.default_rng(sim_rng.integers(0, 2**31))
        pvals = run_single_simulation(data, n_bootstrap=n_bootstrap, rng=boot_rng)
        for m in ALL_METRICS:
            p_values_by_metric[m].append(pvals[m])

    results = []
    for m in ALL_METRICS:
        pv = np.array(p_values_by_metric[m])
        results.append({
            "metric": m,
            "sigma_pert": sigma_pert,
            "sigma": sigma,
            "condition": condition,
            "detection_rate": float(np.mean(pv < 0.05)),
            "mean_p_value": float(np.mean(pv)),
        })
    return results


def run_power_analysis_v2(
    z_i: np.ndarray,
    y_orig: np.ndarray,
    condition: str,
    sigma_pert_values: list[float],
    sigma_values: list[float],
    n_simulations: int = 1000,
    n_bootstrap: int = 1000,
    seed: int = 42,
    n_workers: int = 1,
    checkpoint_path: str = None,
) -> pd.DataFrame:
    """Run Monte Carlo simulation across (sigma_pert, sigma) grid for one condition.

    Uses real logits z_i and binary answers y_orig. Parallelizes across
    parameter combos using n_workers processes. Checkpoints to CSV.

    Returns DataFrame with columns: metric, sigma_pert, sigma, condition,
    detection_rate, mean_p_value.
    """
    import multiprocessing

    completed = set()
    all_results = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        all_results = existing.to_dict('records')
        for _, row in existing.iterrows():
            completed.add((row['sigma_pert'], row['sigma'], row['metric']))
        n_done = len(completed) // len(ALL_METRICS)
        print(f"  Resuming from checkpoint: {n_done} combos already done", flush=True)

    work_items = []
    for sigma_pert in sigma_pert_values:
        for sigma in sigma_values:
            if (sigma_pert, sigma, ALL_METRICS[0]) in completed:
                continue
            work_items.append((
                sigma_pert, sigma, condition,
                n_simulations, n_bootstrap, seed, z_i, y_orig,
            ))

    total = len(work_items)
    print(f"  {total} combos remaining, using {n_workers} workers", flush=True)

    if total == 0:
        return pd.DataFrame(all_results)

    done = 0
    if n_workers <= 1:
        for item in work_items:
            combo_results = _run_one_combo_v2(item)
            all_results.extend(combo_results)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)", flush=True)
                if checkpoint_path:
                    pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)
    else:
        with multiprocessing.Pool(n_workers) as pool:
            for combo_results in pool.imap_unordered(_run_one_combo_v2, work_items):
                all_results.extend(combo_results)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)", flush=True)
                    if checkpoint_path:
                        pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    if checkpoint_path:
        pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(all_results)


def run_power_analysis(
    beta1_values: list[float],
    beta_gender_values: list[float],
    n_simulations: int,
    n_cases: int = 100,
    n_bootstrap: int = 1000,
    beta0: float = 0.0,
    seed: int = 42,
    sigma_values: list[float] = None,
    sigma_pert: float = None,
    n_workers: int = 1,
    checkpoint_path: str = None,
) -> pd.DataFrame:
    """Run Monte Carlo simulation across a parameter grid for all 5 metrics.

    For each (beta1, sigma, beta_gender) combination, runs n_simulations
    synthetic experiments and records how often each metric test rejects
    at p < 0.05. Parallelizes across parameter combos using n_workers processes.

    sigma_pert is the perturbation arm noise (defaults to sigma if not specified).

    Returns DataFrame with columns: metric, beta1, sigma, beta_gender,
    detection_rate, mean_p_value.
    """
    import multiprocessing

    if sigma_values is None:
        sigma_values = [0.0]

    # Load checkpoint if exists
    completed = set()
    all_results = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        all_results = existing.to_dict('records')
        for _, row in existing.iterrows():
            completed.add((row['beta1'], row['sigma'], row['beta_gender'], row['metric']))
        n_combos_done = len(completed) // len(ALL_METRICS)
        print(f"  Resuming from checkpoint: {n_combos_done} combos already done",
              flush=True)

    # Build work items, skipping completed combos
    work_items = []
    for beta1 in beta1_values:
        for sigma in sigma_values:
            for beta_gender in beta_gender_values:
                if (beta1, sigma, beta_gender, ALL_METRICS[0]) in completed:
                    continue
                work_items.append((
                    beta1, sigma, beta_gender,
                    n_simulations, n_cases, n_bootstrap, beta0, seed, sigma_pert,
                ))

    total = len(work_items)
    print(f"  {total} combos remaining, using {n_workers} workers", flush=True)

    if total == 0:
        return pd.DataFrame(all_results)

    done = 0
    with multiprocessing.Pool(n_workers) as pool:
        for combo_results in pool.imap_unordered(_run_one_combo, work_items):
            all_results.extend(combo_results)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  Progress: {done}/{total} combos "
                      f"({done/total*100:.0f}%)", flush=True)
                if checkpoint_path:
                    pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    # Final save
    if checkpoint_path:
        pd.DataFrame(all_results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(all_results)


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


def generate_power_curves_v2(results: pd.DataFrame, output_dir: str, condition: str) -> None:
    """Generate power curve figure for one condition.

    Layout: one subplot per sigma value, 5 metric curves per subplot.
    """
    sigma_values = sorted(results["sigma"].unique())
    n_cols = min(len(sigma_values), 2)
    n_rows = (len(sigma_values) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    metric_colors = {
        "jsd": "#1f77b4", "kl": "#ff7f0e",
        "mi": "#2ca02c", "phi": "#d62728", "flip_rate": "#9467bd",
    }

    for idx, sigma in enumerate(sigma_values):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        subset = results[results["sigma"] == sigma]

        for metric in ["jsd", "kl", "mi", "phi", "flip_rate"]:
            m_data = subset[subset["metric"] == metric].sort_values("sigma_pert")
            ax.plot(
                m_data["sigma_pert"], m_data["detection_rate"],
                marker='o', markersize=3, color=metric_colors[metric],
                label=metric.upper() if metric != "flip_rate" else "Flip Rate",
            )

        ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=0.80, color='lightgray', linestyle=':', linewidth=1)
        ax.set_xlabel(r'$\sigma_{pert}$')
        ax.set_ylabel('Detection rate')
        ax.set_title(f'{condition} ($\\sigma$={sigma})')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    for idx in range(len(sigma_values), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'power_curves_{condition}.png'), dpi=300)
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
    parser.add_argument('--sigma-pert', type=float, default=None,
                        help='Perturbation arm noise level (defaults to sigma)')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str,
                        default='results/simulation',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    beta_gender_values = list(np.arange(0, args.beta_gender_max + 1e-9,
                                        args.beta_gender_step))

    total_combos = (len(args.beta1_values) * len(beta_gender_values)
                    * len(args.sigma_values))
    print(f"Running power analysis:", flush=True)
    print(f"  beta1 values: {args.beta1_values}", flush=True)
    print(f"  sigma values: {args.sigma_values}", flush=True)
    print(f"  beta_gender range: 0 to {args.beta_gender_max} "
          f"(step {args.beta_gender_step})", flush=True)
    print(f"  {args.n_simulations} simulations per combination", flush=True)
    print(f"  {args.n_cases} cases, {args.n_bootstrap} bootstrap iterations",
          flush=True)
    print(f"  Total parameter combos: {total_combos}", flush=True)
    print(f"  Workers: {args.n_workers}", flush=True)
    print(f"  Metrics: {ALL_METRICS}", flush=True)

    checkpoint_path = os.path.join(args.output_dir, 'simulation_results.csv')

    results = run_power_analysis(
        beta1_values=args.beta1_values,
        beta_gender_values=beta_gender_values,
        sigma_values=args.sigma_values,
        sigma_pert=args.sigma_pert,
        n_simulations=args.n_simulations,
        n_cases=args.n_cases,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        n_workers=args.n_workers,
        checkpoint_path=checkpoint_path,
    )

    results_path = os.path.join(args.output_dir, 'simulation_results.csv')
    results.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}", flush=True)

    generate_power_curves(results, args.output_dir)
    print(f"Power curves saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
