# ABOUTME: Tests for evaluate_neutral_baseline.py
# ABOUTME: Verifies neutral baseline evaluation filtering, sharding, and atomic saves

import pytest
import pandas as pd
import json
import os
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')


class TestGetNeutralBaselineRows:
    """Tests for get_neutral_baseline_rows filtering."""

    def test_filters_to_dataset_id_6(self):
        """Must return only rows with dataset_id=6."""
        from evaluate_neutral_baseline import get_neutral_baseline_rows

        df = pd.DataFrame({
            'Index': list(range(10)),
            'dataset_id': [1, 2, 3, 4, 5, 6, 6, 8, 9, 1],
            'clinical_context': [f'text {i}' for i in range(10)]
        })

        result = get_neutral_baseline_rows(df)

        assert len(result) == 2
        assert (result['dataset_id'] == 6).all()
        assert list(result['Index']) == [5, 6]

    def test_returns_copy(self):
        """Modification of returned df must not affect the original."""
        from evaluate_neutral_baseline import get_neutral_baseline_rows

        df = pd.DataFrame({
            'Index': [0, 1, 2],
            'dataset_id': [6, 6, 1],
            'clinical_context': ['a', 'b', 'c']
        })

        result = get_neutral_baseline_rows(df)
        result['clinical_context'] = 'modified'

        assert df.loc[0, 'clinical_context'] == 'a'
        assert df.loc[1, 'clinical_context'] == 'b'


class TestShardData:
    """Tests for shard_data distribution."""

    def test_shards_evenly(self):
        """Must distribute rows evenly across GPUs via interleaving."""
        from evaluate_neutral_baseline import shard_data

        df = pd.DataFrame({
            'Index': list(range(12)),
            'clinical_context': [f'text {i}' for i in range(12)]
        })

        shard0 = shard_data(df, gpu_id=0, total_gpus=3)
        shard1 = shard_data(df, gpu_id=1, total_gpus=3)
        shard2 = shard_data(df, gpu_id=2, total_gpus=3)

        assert len(shard0) == 4
        assert len(shard1) == 4
        assert len(shard2) == 4
        assert list(shard0['Index']) == [0, 3, 6, 9]
        assert list(shard1['Index']) == [1, 4, 7, 10]
        assert list(shard2['Index']) == [2, 5, 8, 11]


class TestAtomicSaveCsv:
    """Tests for atomic_save_csv."""

    def test_atomic_save_csv(self, tmp_path):
        """File must be written with no temp files left behind."""
        from evaluate_neutral_baseline import atomic_save_csv

        output_path = str(tmp_path / 'checkpoint.csv')
        df = pd.DataFrame({
            'Index': [800, 801],
            'LLAMA3_MANAGE': [1, 0],
        })

        atomic_save_csv(df, output_path)

        assert os.path.exists(output_path)
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 2
        assert list(loaded['Index']) == [800, 801]

        temp_files = [f for f in os.listdir(tmp_path) if f.endswith('.tmp')]
        assert len(temp_files) == 0, f"Temp files left behind: {temp_files}"


class TestAtomicSaveJson:
    """Tests for atomic_save_json."""

    def test_atomic_save_json(self, tmp_path):
        """File must be written with no temp files left behind."""
        from evaluate_neutral_baseline import atomic_save_json

        output_path = str(tmp_path / 'trace.json')
        data = [{"Index": 800, "MANAGE": {"binary_answers": [1, 1, 0]}}]

        atomic_save_json(data, output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded == data

        temp_files = [f for f in os.listdir(tmp_path) if f.endswith('.tmp')]
        assert len(temp_files) == 0, f"Temp files left behind: {temp_files}"


class TestMajorityVote:
    """Tests for majority_vote."""

    def test_majority_yes(self):
        """[1, 1, 0] must return 1."""
        from evaluate_neutral_baseline import majority_vote
        assert majority_vote([1, 1, 0]) == 1

    def test_majority_no(self):
        """[0, 0, 1] must return 0."""
        from evaluate_neutral_baseline import majority_vote
        assert majority_vote([0, 0, 1]) == 0

    def test_unanimous_yes(self):
        """[1, 1, 1] must return 1."""
        from evaluate_neutral_baseline import majority_vote
        assert majority_vote([1, 1, 1]) == 1

    def test_unanimous_no(self):
        """[0, 0, 0] must return 0."""
        from evaluate_neutral_baseline import majority_vote
        assert majority_vote([0, 0, 0]) == 0


class TestGetColumnPrefix:
    """Tests for get_column_prefix model name mapping."""

    def test_llama3_8b(self):
        """Llama-3 8B models must map to 'LLAMA3'."""
        from evaluate_neutral_baseline import get_column_prefix
        assert get_column_prefix('meta-llama/Llama-3.1-8B-Instruct') == 'LLAMA3'

    def test_llama3_70b(self):
        """Llama-3 70B models must map to 'LLAMA3-70'."""
        from evaluate_neutral_baseline import get_column_prefix
        assert get_column_prefix('meta-llama/Llama-3.3-70B-Instruct') == 'LLAMA3-70'

    def test_unknown_model_raises(self):
        """Unknown model names must raise ValueError."""
        from evaluate_neutral_baseline import get_column_prefix
        with pytest.raises(ValueError, match="Unknown model"):
            get_column_prefix('some-unknown-model')
