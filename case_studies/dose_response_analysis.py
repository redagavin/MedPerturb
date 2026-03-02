# ABOUTME: Analyze dose-response relationship between token change % and answer instability
# ABOUTME: Computes flip rate and MI at each level, generates plots with bootstrap error bars

import numpy as np
import pandas as pd
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def majority_vote(responses):
    """Return majority vote from list of 0/1 responses."""
    return 1 if sum(responses) > len(responses) / 2 else 0


def compute_flip_rate(orig_votes, para_votes):
    """Compute proportion of samples where answer flipped.

    Args:
        orig_votes: List of binary original answers
        para_votes: List of binary paraphrase answers

    Returns:
        float: Proportion of flipped answers
    """
    n = len(orig_votes)
    if n == 0:
        return 0.0
    flips = sum(1 for o, p in zip(orig_votes, para_votes) if o != p)
    return flips / n


def bootstrap_flip_rate_se(orig_votes, para_votes, n_bootstrap=1000):
    """Compute bootstrap SE of flip rate.

    Args:
        orig_votes: List of binary original answers
        para_votes: List of binary paraphrase answers
        n_bootstrap: Number of bootstrap iterations

    Returns:
        float: Standard error of flip rate
    """
    n = len(orig_votes)
    orig_arr = np.array(orig_votes)
    para_arr = np.array(para_votes)
    rates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        flips = np.sum(orig_arr[idx] != para_arr[idx])
        rates.append(flips / n)
    return np.std(rates)


def calculate_mi(x, y):
    """Calculate mutual information between two binary arrays.

    Uses pd.crosstab-based algorithm matching baseline_analysis.py.

    Args:
        x: List/array of binary values
        y: List/array of binary values

    Returns:
        float: Mutual information in bits
    """
    x_s = pd.Series(x)
    y_s = pd.Series(y)
    joint = pd.crosstab(x_s, y_s, normalize=True)
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    mi = 0.0
    for i in joint.index:
        for j in joint.columns:
            if joint.loc[i, j] > 0:
                mi += joint.loc[i, j] * np.log2(
                    joint.loc[i, j] / (p_x[i] * p_y[j])
                )
    return mi


def bootstrap_mi_se(x, y, n_bootstrap=1000):
    """Compute bootstrap SE of mutual information.

    Args:
        x: List/array of binary values
        y: List/array of binary values
        n_bootstrap: Number of bootstrap iterations

    Returns:
        float: Standard error of MI
    """
    n = len(x)
    x_arr = np.array(x)
    y_arr = np.array(y)
    mis = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        mi = calculate_mi(x_arr[idx].tolist(), y_arr[idx].tolist())
        mis.append(mi)
    return np.std(mis)
