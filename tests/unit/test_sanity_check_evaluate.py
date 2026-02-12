# ABOUTME: Tests for sanity check evaluation data loading and alignment
# ABOUTME: Verifies correct filtering of non-conversational samples and context_id alignment

import pytest
import pandas as pd


@pytest.fixture
def sample_dataset_csv(tmp_path):
    """Create a minimal data_with_baselines.csv for testing."""
    rows = [
        # Originals (dataset_id=1)
        {"Index": 0, "dataset": "askadoc", "dataset_id": 1, "context_id": "N75",
         "clinical_context": "19 year old white female with iron deficiency"},
        {"Index": 1, "dataset": "askadoc", "dataset_id": 1, "context_id": "N70",
         "clinical_context": "24M presents with vomiting after drinking"},
        {"Index": 2, "dataset": "conversational", "dataset_id": 1, "context_id": "C1",
         "clinical_context": "conversational original text"},
        # Gender swaps (dataset_id=2)
        {"Index": 3, "dataset": "askadoc", "dataset_id": 2, "context_id": "N75",
         "clinical_context": "19 year old white male with iron deficiency"},
        {"Index": 4, "dataset": "askadoc", "dataset_id": 2, "context_id": "N70",
         "clinical_context": "24F presents with vomiting after drinking"},
        {"Index": 5, "dataset": "conversational", "dataset_id": 2, "context_id": "C1",
         "clinical_context": "conversational swap text"},
        # Gender swap baselines (dataset_id=6)
        {"Index": 6, "dataset": "askadoc", "dataset_id": 6, "context_id": "N75",
         "clinical_context": "A 19yo white female who is iron deficient"},
        {"Index": 7, "dataset": "askadoc", "dataset_id": 6, "context_id": "N70",
         "clinical_context": "24 year old male with post-drinking vomiting"},
        {"Index": 8, "dataset": "conversational", "dataset_id": 6, "context_id": "C1",
         "clinical_context": "conversational baseline text"},
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestLoadSanityCheckData:
    """Tests for loading and aligning sanity check data."""

    def test_excludes_conversational(self, sample_dataset_csv):
        """Conversational subset must be excluded."""
        from sanity_check_evaluate import load_sanity_check_data
        samples = load_sanity_check_data(sample_dataset_csv)
        context_ids = [s['context_id'] for s in samples]
        assert "C1" not in context_ids

    def test_returns_non_conversational_only(self, sample_dataset_csv):
        """Should return exactly the non-conversational samples."""
        from sanity_check_evaluate import load_sanity_check_data
        samples = load_sanity_check_data(sample_dataset_csv)
        assert len(samples) == 2
        context_ids = {s['context_id'] for s in samples}
        assert context_ids == {"N75", "N70"}

    def test_aligns_three_versions_by_context_id(self, sample_dataset_csv):
        """Each sample must have original, swap, and baseline text."""
        from sanity_check_evaluate import load_sanity_check_data
        samples = load_sanity_check_data(sample_dataset_csv)
        for s in samples:
            assert 'original_text' in s
            assert 'swap_text' in s
            assert 'baseline_text' in s
            assert s['original_text'] != ''
            assert s['swap_text'] != ''
            assert s['baseline_text'] != ''

    def test_correct_text_alignment(self, sample_dataset_csv):
        """Verify texts are correctly matched to their versions."""
        from sanity_check_evaluate import load_sanity_check_data
        samples = load_sanity_check_data(sample_dataset_csv)
        n75 = [s for s in samples if s['context_id'] == 'N75'][0]
        assert "female" in n75['original_text']
        assert "male" in n75['swap_text']
        assert "female" in n75['baseline_text']  # baseline preserves original gender

    def test_only_includes_samples_with_all_three_versions(self, sample_dataset_csv, tmp_path):
        """If a context_id is missing any version, exclude it."""
        # Create dataset where N70 has no baseline
        rows = [
            {"Index": 0, "dataset": "askadoc", "dataset_id": 1, "context_id": "N75",
             "clinical_context": "19 year old white female"},
            {"Index": 1, "dataset": "askadoc", "dataset_id": 1, "context_id": "N70",
             "clinical_context": "24M with vomiting"},
            {"Index": 2, "dataset": "askadoc", "dataset_id": 2, "context_id": "N75",
             "clinical_context": "19 year old white male"},
            {"Index": 3, "dataset": "askadoc", "dataset_id": 2, "context_id": "N70",
             "clinical_context": "24F with vomiting"},
            # Only N75 has a baseline
            {"Index": 4, "dataset": "askadoc", "dataset_id": 6, "context_id": "N75",
             "clinical_context": "A 19yo white female"},
        ]
        df = pd.DataFrame(rows)
        path = tmp_path / "incomplete_data.csv"
        df.to_csv(path, index=False)

        from sanity_check_evaluate import load_sanity_check_data
        samples = load_sanity_check_data(str(path))
        assert len(samples) == 1
        assert samples[0]['context_id'] == 'N75'
