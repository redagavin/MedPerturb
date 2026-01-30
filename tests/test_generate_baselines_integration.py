# ABOUTME: Integration tests for baseline generation main pipeline
# ABOUTME: Validates perturbation type constants and dataset loading

import pytest
import pandas as pd
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')

from generate_baselines import load_dataset_with_perturbations, PERTURBATION_TYPES


def test_perturbation_types():
    """Verify we have the 4 expected perturbation types."""
    expected = ['gender_swap', 'gender_remove', 'stylistic_uncertain', 'stylistic_colorful']
    assert PERTURBATION_TYPES == expected


def test_load_dataset_columns():
    """Dataset must have Index (unique row ID) and clinical_context columns."""
    df = pd.read_csv('/scratch/yang.zih/cot_faithfulness/MedPerturb/data.csv')
    assert 'Index' in df.columns, "Dataset missing 'Index' column"
    assert 'clinical_context' in df.columns, "Dataset missing 'clinical_context' column"
    assert df['Index'].nunique() == len(df), (
        f"Index is not unique: {df['Index'].nunique()} unique vs {len(df)} rows"
    )
