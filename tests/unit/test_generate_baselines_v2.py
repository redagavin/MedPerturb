# ABOUTME: Tests for generate_baselines_v2.py with correct dataset_id/context_id data flow
# ABOUTME: Verifies baseline generation uses perturbation rows to find originals

import pytest
import pandas as pd


class TestFindOriginal:
    """Tests for finding original row given a perturbation row."""

    def test_find_original_returns_matching_row(self):
        """find_original must return row with same dataset+context_id and dataset_id=1."""
        from generate_baselines_v2 import find_original

        df = pd.DataFrame({
            'Index': [0, 1, 2, 3],
            'dataset': ['askadoc', 'askadoc', 'askadoc', 'askadoc'],
            'dataset_id': [1, 2, 1, 2],
            'context_id': ['N75', 'N75', 'N70', 'N70'],
            'clinical_context': ['orig A', 'pert A', 'orig B', 'pert B']
        })

        # Find original for perturbation at Index=1
        pert_row = df[df['Index'] == 1].iloc[0]
        original = find_original(df, pert_row)

        assert original['Index'] == 0
        assert original['dataset_id'] == 1
        assert original['context_id'] == 'N75'
        assert original['clinical_context'] == 'orig A'

    def test_find_original_handles_different_datasets(self):
        """find_original must match on dataset field, not just context_id."""
        from generate_baselines_v2 import find_original

        # Same context_id but different dataset sources
        df = pd.DataFrame({
            'Index': [0, 1, 2, 3],
            'dataset': ['askadoc', 'askadoc', 'oncqa', 'oncqa'],
            'dataset_id': [1, 2, 1, 2],
            'context_id': ['33', '33', '33', '33'],  # Same context_id, different dataset
            'clinical_context': ['askadoc orig', 'askadoc pert', 'oncqa orig', 'oncqa pert']
        })

        # Find original for oncqa perturbation
        pert_row = df[df['Index'] == 3].iloc[0]
        original = find_original(df, pert_row)

        assert original['dataset'] == 'oncqa'
        assert original['clinical_context'] == 'oncqa orig'

    def test_find_original_raises_for_missing(self):
        """find_original must raise KeyError if no matching original exists."""
        from generate_baselines_v2 import find_original

        df = pd.DataFrame({
            'Index': [0, 1],
            'dataset': ['askadoc', 'oncqa'],
            'dataset_id': [1, 2],
            'context_id': ['N75', 'N99'],
            'clinical_context': ['orig', 'pert without orig']
        })

        pert_row = df[df['Index'] == 1].iloc[0]

        with pytest.raises(KeyError):
            find_original(df, pert_row)
