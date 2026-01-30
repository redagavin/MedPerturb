# ABOUTME: Tests for create_extended_dataset.py
# ABOUTME: Verifies correct merging of baselines into dataset with proper schema

import pytest
import pandas as pd
import json
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/.worktrees/calibrated-baselines/code')


class TestCreateExtendedDataset:
    """Tests for creating extended dataset with baselines."""

    def test_appends_baseline_rows(self, tmp_path):
        """Must append baseline rows to original dataset."""
        from create_extended_dataset import create_extended_dataset

        # Create minimal original dataset
        original_df = pd.DataFrame({
            'Index': [0, 1],
            'dataset': ['askadoc', 'askadoc'],
            'dataset_id': [1, 2],
            'context_id': ['N75', 'N75'],
            'original_patient_gender': ['F', 'F'],
            'clinical_context': ['original text', 'perturbed text'],
            'LLAMA3_MANAGE': [1, 0],
            'LLAMA3_VISIT': [0, 1],
            'LLAMA3_RESOURCE': [1, 1],
        })

        # Create baselines
        baselines = [{
            'source_index': 1,
            'original_index': 0,
            'dataset': 'askadoc',
            'context_id': 'N75',
            'perturbation_type': 2,
            'baseline_dataset_id': 6,
            'paraphrase': 'baseline paraphrase text',
            'target_pct': 15.0,
            'actual_pct': 14.8,
            'deviation': 0.2,
            'retries_used': 1
        }]

        dataset_path = tmp_path / 'data.csv'
        baselines_path = tmp_path / 'baselines.json'
        output_path = tmp_path / 'data_with_baselines.csv'

        original_df.to_csv(dataset_path, index=False)
        with open(baselines_path, 'w') as f:
            json.dump(baselines, f)

        result_df = create_extended_dataset(
            str(dataset_path),
            str(baselines_path),
            str(output_path)
        )

        # Should have 3 rows: 2 original + 1 baseline
        assert len(result_df) == 3

        # Baseline row should have correct values
        baseline_row = result_df[result_df['dataset_id'] == 6].iloc[0]
        assert baseline_row['Index'] == 800  # Baselines start at 800
        assert baseline_row['clinical_context'] == 'baseline paraphrase text'
        assert baseline_row['dataset'] == 'askadoc'
        assert baseline_row['context_id'] == 'N75'

    def test_baseline_index_starts_at_800(self, tmp_path):
        """Baseline Index values must start at 800."""
        from create_extended_dataset import create_extended_dataset

        # Create dataset with indices 0-799
        original_df = pd.DataFrame({
            'Index': list(range(800)),
            'dataset': ['askadoc'] * 800,
            'dataset_id': [1] * 200 + [2] * 200 + [3] * 200 + [4] * 100 + [5] * 100,
            'context_id': [f'N{i}' for i in range(800)],
            'original_patient_gender': ['F'] * 800,
            'clinical_context': [f'text {i}' for i in range(800)],
        })

        baselines = [{
            'source_index': 200,
            'original_index': 0,
            'dataset': 'askadoc',
            'context_id': 'N0',
            'perturbation_type': 2,
            'baseline_dataset_id': 6,
            'paraphrase': 'baseline text',
            'target_pct': 10.0,
            'actual_pct': 10.0,
            'deviation': 0.0,
            'retries_used': 0
        }]

        dataset_path = tmp_path / 'data.csv'
        baselines_path = tmp_path / 'baselines.json'
        output_path = tmp_path / 'data_with_baselines.csv'

        original_df.to_csv(dataset_path, index=False)
        with open(baselines_path, 'w') as f:
            json.dump(baselines, f)

        result_df = create_extended_dataset(
            str(dataset_path),
            str(baselines_path),
            str(output_path)
        )

        baseline_row = result_df[result_df['dataset_id'] == 6].iloc[0]
        assert baseline_row['Index'] == 800

    def test_model_columns_empty_for_baselines(self, tmp_path):
        """Model output columns must be empty/NaN for baseline rows."""
        from create_extended_dataset import create_extended_dataset

        original_df = pd.DataFrame({
            'Index': [0, 1],
            'dataset': ['askadoc', 'askadoc'],
            'dataset_id': [1, 2],
            'context_id': ['N75', 'N75'],
            'original_patient_gender': ['F', 'F'],
            'clinical_context': ['orig', 'pert'],
            'LLAMA3_MANAGE': [1, 0],
            'LLAMA3-70_MANAGE': [1, 1],
        })

        baselines = [{
            'source_index': 1,
            'original_index': 0,
            'dataset': 'askadoc',
            'context_id': 'N75',
            'perturbation_type': 2,
            'baseline_dataset_id': 6,
            'paraphrase': 'baseline',
            'target_pct': 10.0,
            'actual_pct': 10.0,
            'deviation': 0.0,
            'retries_used': 0
        }]

        dataset_path = tmp_path / 'data.csv'
        baselines_path = tmp_path / 'baselines.json'
        output_path = tmp_path / 'data_with_baselines.csv'

        original_df.to_csv(dataset_path, index=False)
        with open(baselines_path, 'w') as f:
            json.dump(baselines, f)

        result_df = create_extended_dataset(
            str(dataset_path),
            str(baselines_path),
            str(output_path)
        )

        baseline_row = result_df[result_df['dataset_id'] == 6].iloc[0]
        assert pd.isna(baseline_row['LLAMA3_MANAGE'])
        assert pd.isna(baseline_row['LLAMA3-70_MANAGE'])
