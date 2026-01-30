# ABOUTME: Generate calibrated baselines for MedPerturb with correct data flow
# ABOUTME: Uses dataset_id/context_id structure to link perturbations to originals

import pandas as pd


def find_original(df: pd.DataFrame, pert_row: pd.Series) -> pd.Series:
    """
    Find the original row corresponding to a perturbation row.

    Args:
        df: Full dataset DataFrame
        pert_row: A perturbation row (dataset_id in 2,3,4,5)

    Returns:
        pd.Series: The matching original row (dataset_id=1)

    Raises:
        KeyError: If no matching original found
    """
    matches = df[
        (df['dataset'] == pert_row['dataset']) &
        (df['context_id'] == pert_row['context_id']) &
        (df['dataset_id'] == 1)
    ]

    if len(matches) == 0:
        raise KeyError(
            f"No original found for dataset={pert_row['dataset']}, "
            f"context_id={pert_row['context_id']}"
        )

    return matches.iloc[0]
