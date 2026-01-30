# ABOUTME: Tests for evaluate_baselines.py
# ABOUTME: Verifies baseline-only evaluation with correct model output columns

import pytest
import pandas as pd
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/.worktrees/calibrated-baselines/code')


class TestEvaluateBaselines:
    """Tests for baseline evaluation."""

    def test_only_evaluates_baseline_rows(self):
        """Must only evaluate rows with dataset_id 6-9."""
        from evaluate_baselines import get_baseline_rows

        df = pd.DataFrame({
            'Index': list(range(10)),
            'dataset_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
            'clinical_context': [f'text {i}' for i in range(10)]
        })

        baseline_df = get_baseline_rows(df)

        assert len(baseline_df) == 4
        assert set(baseline_df['dataset_id'].unique()) == {6, 7, 8, 9}

    def test_shard_data_for_gpu(self):
        """Must correctly shard data across GPUs."""
        from evaluate_baselines import shard_data

        df = pd.DataFrame({
            'Index': list(range(100)),
            'clinical_context': [f'text {i}' for i in range(100)]
        })

        # GPU 0 of 4
        shard0 = shard_data(df, gpu_id=0, total_gpus=4)
        assert len(shard0) == 25
        assert list(shard0['Index']) == list(range(0, 100, 4))

        # GPU 1 of 4
        shard1 = shard_data(df, gpu_id=1, total_gpus=4)
        assert len(shard1) == 25
        assert list(shard1['Index']) == list(range(1, 100, 4))

    def test_model_name_to_column_prefix(self):
        """Must map model names to correct column prefixes."""
        from evaluate_baselines import get_column_prefix

        assert get_column_prefix('meta-llama/Llama-3.1-8B-Instruct') == 'LLAMA3'
        assert get_column_prefix('meta-llama/Llama-3.3-70B-Instruct') == 'LLAMA3-70'
