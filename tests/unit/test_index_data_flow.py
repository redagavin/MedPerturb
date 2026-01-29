# ABOUTME: Tests that Index-based joins work correctly across pipeline stages
# ABOUTME: Verifies load_all_data() correctly maps perturbations and baselines by Index

import json
import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')

from batch_evaluate import load_all_data, PERTURBATION_TYPES


@pytest.fixture
def pipeline_data(tmp_path):
    """Create a small dataset, perturbation JSONs, and baselines JSON with known Index values."""
    # Dataset with non-sequential Index values to catch off-by-one or positional bugs
    df = pd.DataFrame({
        'Index': [10, 20, 30],
        'clinical_context': ['Patient A context', 'Patient B context', 'Patient C context'],
        'context_id': ['case_a', 'case_b', 'case_c'],
        'dataset': ['askadocs', 'askadocs', 'askadocs'],
    })
    csv_path = tmp_path / 'data.csv'
    df.to_csv(csv_path, index=False)

    # Perturbation files — only gender_swap has data, others missing
    perturbations_dir = tmp_path / 'perturbations'
    perturbations_dir.mkdir()

    gender_swap = {
        '10': {'original': 'Patient A context', 'perturbed': 'Patient A gender swapped'},
        '20': {'original': 'Patient B context', 'perturbed': 'Patient B gender swapped'},
        '30': {'original': 'Patient C context', 'perturbed': 'Patient C gender swapped'},
    }
    with open(perturbations_dir / 'gender_swap.json', 'w') as f:
        json.dump(gender_swap, f)

    # Baselines
    baselines = {
        '10': {'gender_swap': {'paraphrase': 'Patient A baseline for gender_swap'}},
        '20': {'gender_swap': {'paraphrase': 'Patient B baseline for gender_swap'}},
        '30': {'gender_swap': {'paraphrase': 'Patient C baseline for gender_swap'}},
    }
    baselines_path = tmp_path / 'baselines.json'
    with open(baselines_path, 'w') as f:
        json.dump(baselines, f)

    return str(csv_path), str(perturbations_dir), str(baselines_path)


def test_load_all_data_joins_by_index(pipeline_data):
    """Perturbations and baselines must be joined to the correct row by Index."""
    csv_path, perturbations_dir, baselines_path = pipeline_data
    df = load_all_data(csv_path, perturbations_dir, baselines_path)

    # Verify each row got the right perturbation by checking Index → perturbed text
    row_10 = df[df['Index'] == 10].iloc[0]
    row_20 = df[df['Index'] == 20].iloc[0]
    row_30 = df[df['Index'] == 30].iloc[0]

    assert row_10['gender_swap_perturbed'] == 'Patient A gender swapped'
    assert row_20['gender_swap_perturbed'] == 'Patient B gender swapped'
    assert row_30['gender_swap_perturbed'] == 'Patient C gender swapped'

    assert row_10['gender_swap_baseline'] == 'Patient A baseline for gender_swap'
    assert row_20['gender_swap_baseline'] == 'Patient B baseline for gender_swap'
    assert row_30['gender_swap_baseline'] == 'Patient C baseline for gender_swap'


def test_missing_perturbation_file_yields_empty(pipeline_data):
    """Missing perturbation JSON files should produce empty strings, not errors."""
    csv_path, perturbations_dir, baselines_path = pipeline_data
    df = load_all_data(csv_path, perturbations_dir, baselines_path)

    # gender_remove.json doesn't exist — should be empty
    assert (df['gender_remove_perturbed'] == '').all()


def test_missing_index_in_perturbation_yields_empty(tmp_path):
    """If a perturbation file is missing an Index, that row should get empty string."""
    df = pd.DataFrame({
        'Index': [1, 2],
        'clinical_context': ['ctx1', 'ctx2'],
        'context_id': ['c1', 'c2'],
        'dataset': ['askadocs', 'askadocs'],
    })
    csv_path = tmp_path / 'data.csv'
    df.to_csv(csv_path, index=False)

    perturbations_dir = tmp_path / 'perturbations'
    perturbations_dir.mkdir()

    # Only Index 1 has a perturbation, Index 2 is missing
    gender_swap = {
        '1': {'original': 'ctx1', 'perturbed': 'ctx1 swapped'},
    }
    with open(perturbations_dir / 'gender_swap.json', 'w') as f:
        json.dump(gender_swap, f)

    baselines_path = tmp_path / 'baselines.json'
    with open(baselines_path, 'w') as f:
        json.dump({}, f)

    result = load_all_data(str(csv_path), str(perturbations_dir), str(baselines_path))

    assert result[result['Index'] == 1].iloc[0]['gender_swap_perturbed'] == 'ctx1 swapped'
    assert result[result['Index'] == 2].iloc[0]['gender_swap_perturbed'] == ''
