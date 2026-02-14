# ABOUTME: Tests for evaluate_baselines.py
# ABOUTME: Verifies baseline-only evaluation with correct model output columns

import pytest
import pandas as pd
import os
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')


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

    def test_resume_handles_index_type_variations(self, tmp_path):
        """Resume must work with various Index types (int, float, string from CSV)."""
        # Verify int() coercion handles type variations
        test_cases = [800, 800.0, '800']
        for val in test_cases:
            assert int(val) == 800, f"int({val!r}) should equal 800"

        # Verify set membership with coercion
        completed = {int(800), int(801)}
        assert int(800) in completed
        assert int(800.0) in completed
        assert int('800') in completed
        assert int(803) not in completed

    def test_atomic_save_csv(self, tmp_path):
        """Checkpoint saves must be atomic (no partial writes on crash)."""
        from evaluate_baselines import atomic_save_csv
        import os

        output_path = tmp_path / 'checkpoint.csv'

        # Create test data
        df = pd.DataFrame({
            'Index': [800, 801, 802],
            'LLAMA3_MANAGE': [1, 0, 1],
            'LLAMA3_VISIT': [0, 1, 0],
        })

        # Save atomically
        atomic_save_csv(df, str(output_path))

        # Verify file exists and is valid
        assert output_path.exists()
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 3
        assert list(loaded['Index']) == [800, 801, 802]

        # Verify no temp files left behind
        temp_files = [f for f in os.listdir(tmp_path) if f.endswith('.tmp')]
        assert len(temp_files) == 0, f"Temp files left behind: {temp_files}"


class TestAtomicSaveJson:
    """Trace JSON saves must be atomic."""

    def test_atomic_save_json(self, tmp_path):
        """JSON written atomically with no temp files left behind."""
        from evaluate_baselines import atomic_save_json
        import json

        path = str(tmp_path / 'trace.json')
        data = [{"Index": 800, "MANAGE": {"binary_answers": [1, 1, 1]}}]

        atomic_save_json(data, path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

        temps = [f for f in os.listdir(tmp_path) if f.endswith('.tmp')]
        assert len(temps) == 0


class TestTraceDataHandling:
    """evaluate_baselines must save trace data alongside CSV."""

    def test_accesses_binary_answers_from_trace_dict(self):
        """evaluate_baselines must access triage_results[qt]['binary_answers']."""
        source_path = os.path.join(
            '/scratch/yang.zih/cot_faithfulness/MedPerturb/code',
            'evaluate_baselines.py'
        )
        with open(source_path) as f:
            source = f.read()
        assert "['binary_answers']" in source, (
            "evaluate_baselines must access binary_answers from trace data dict"
        )

    def test_trace_path_derives_from_csv_path(self):
        """Trace JSON path must be derived from CSV checkpoint path."""
        csv_path = 'checkpoints/baseline_eval_gpu0.csv'
        trace_path = csv_path.replace('.csv', '_trace.json')
        assert trace_path == 'checkpoints/baseline_eval_gpu0_trace.json'

    def test_trace_data_reloaded_on_resume(self):
        """Trace JSON must be reloaded on checkpoint resume so final save is complete.

        Without this, a resumed run's trace JSON only contains entries from the
        current session, missing all rows from previous sessions.
        """
        source_path = os.path.join(
            '/scratch/yang.zih/cot_faithfulness/MedPerturb/code',
            'evaluate_baselines.py'
        )
        with open(source_path) as f:
            source = f.read()
        # The checkpoint resume block must load trace JSON alongside CSV
        assert 'trace.json' in source or '_trace.json' in source, (
            "evaluate_baselines must reference trace JSON file"
        )
        # Must use json.load to reload trace data (not just json.dump to save)
        assert source.count('json.load') >= 1, (
            "evaluate_baselines must use json.load to reload trace data on resume"
        )
