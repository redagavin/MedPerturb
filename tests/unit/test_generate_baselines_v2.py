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


import asyncio
import json


class TestGenerateBaselinesForPerturbations:
    """Tests for the main baseline generation function."""

    def test_generates_baseline_for_each_perturbation(self, tmp_path):
        """Must generate one baseline per perturbation row."""
        from generate_baselines_v2 import generate_baselines_for_perturbations

        df = pd.DataFrame({
            'Index': [0, 1, 2],
            'dataset': ['askadoc', 'askadoc', 'askadoc'],
            'dataset_id': [1, 2, 3],
            'context_id': ['N75', 'N75', 'N75'],
            'clinical_context': ['Original text here', 'Gender swapped text', 'Gender removed text']
        })

        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text.split())))

        # Mock API call that returns paraphrase
        async def mock_api(original, target_pct, tokenizer, openai_client, max_retries, tolerance):
            return {
                'paraphrase': f'Paraphrased: {original[:20]}',
                'actual_pct': target_pct,
                'deviation': 0.1,
                'retries_used': 0
            }

        output_path = tmp_path / 'baselines.json'

        baselines = asyncio.run(generate_baselines_for_perturbations(
            df=df,
            tokenizer=MockTokenizer(),
            output_path=str(output_path),
            max_concurrent=10,
            _paraphrase_fn=mock_api
        ))

        # Should have 2 baselines (for dataset_id 2 and 3)
        assert len(baselines) == 2

        # Check structure
        for b in baselines:
            assert 'source_index' in b  # Which perturbation
            assert 'original_index' in b  # Which original
            assert 'baseline_dataset_id' in b  # 6, 7, 8, or 9
            assert 'paraphrase' in b
            assert 'target_pct' in b
            assert 'actual_pct' in b

    def test_baseline_dataset_id_mapping(self, tmp_path):
        """Baseline dataset_id must be perturbation dataset_id + 4."""
        from generate_baselines_v2 import generate_baselines_for_perturbations

        df = pd.DataFrame({
            'Index': [0, 1, 2, 3, 4],
            'dataset': ['askadoc'] * 5,
            'dataset_id': [1, 2, 3, 4, 5],
            'context_id': ['N75'] * 5,
            'clinical_context': ['orig', 'pert2', 'pert3', 'pert4', 'pert5']
        })

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(10))

        async def mock_api(original, target_pct, tokenizer, openai_client, max_retries, tolerance):
            return {'paraphrase': 'p', 'actual_pct': 5.0, 'deviation': 0.1, 'retries_used': 0}

        output_path = tmp_path / 'baselines.json'

        baselines = asyncio.run(generate_baselines_for_perturbations(
            df=df,
            tokenizer=MockTokenizer(),
            output_path=str(output_path),
            max_concurrent=10,
            _paraphrase_fn=mock_api
        ))

        # Check dataset_id mapping: 2->6, 3->7, 4->8, 5->9
        baseline_by_source = {b['source_index']: b for b in baselines}
        assert baseline_by_source[1]['baseline_dataset_id'] == 6  # gender-swap -> baseline-for-gender-swap
        assert baseline_by_source[2]['baseline_dataset_id'] == 7  # gender-remove -> baseline-for-gender-remove
        assert baseline_by_source[3]['baseline_dataset_id'] == 8  # uncertain -> baseline-for-uncertain
        assert baseline_by_source[4]['baseline_dataset_id'] == 9  # colorful -> baseline-for-colorful

    def test_saves_checkpoint_periodically(self, tmp_path):
        """Must save checkpoint during processing."""
        from generate_baselines_v2 import generate_baselines_for_perturbations

        df = pd.DataFrame({
            'Index': list(range(25)),
            'dataset': ['askadoc'] * 25,
            'dataset_id': [1] + [2] * 24,  # 1 original, 24 perturbations
            'context_id': ['N75'] * 25,
            'clinical_context': ['orig'] + [f'pert{i}' for i in range(24)]
        })

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(10))

        checkpoint_calls = []

        async def mock_api(original, target_pct, tokenizer, openai_client, max_retries, tolerance):
            return {'paraphrase': 'p', 'actual_pct': 5.0, 'deviation': 0.1, 'retries_used': 0}

        output_path = tmp_path / 'baselines.json'

        # Monkey-patch json.dump to track checkpoint saves
        original_dump = json.dump
        def tracking_dump(obj, f, **kwargs):
            checkpoint_calls.append(len(obj))
            return original_dump(obj, f, **kwargs)

        json.dump = tracking_dump
        try:
            asyncio.run(generate_baselines_for_perturbations(
                df=df,
                tokenizer=MockTokenizer(),
                output_path=str(output_path),
                max_concurrent=5,
                checkpoint_freq=10,
                _paraphrase_fn=mock_api
            ))
        finally:
            json.dump = original_dump

        # Should have saved checkpoints
        assert len(checkpoint_calls) >= 2, f"Expected at least 2 checkpoints, got {len(checkpoint_calls)}"
