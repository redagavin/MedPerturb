# ABOUTME: Tests for neutral sentence baseline data generation
# ABOUTME: Verifies sentence prepending, filtering, and dataset structure

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')


def _make_test_data():
    """Create test data with askadoc, oncqa, and conversational rows."""
    rows = [
        # askadoc original (dataset_id=1)
        {'Index': 0, 'dataset': 'askadoc', 'dataset_id': 1, 'context_id': 'N1',
         'original_patient_gender': 'F', 'clinical_context': 'I have a headache.',
         'GPT4_MANAGE': 1, 'LLAMA3_MANAGE': 0, 'LLAMA3-70_MANAGE': 1,
         'PALMYRA-MED_MANAGE': 0, 'GPT4_VISIT': 1, 'LLAMA3_VISIT': 0,
         'LLAMA3-70_VISIT': 1, 'PALMYRA-MED_VISIT': 0, 'GPT4_RESOURCE': 1,
         'LLAMA3_RESOURCE': 0, 'LLAMA3-70_RESOURCE': 1, 'PALMYRA-MED_RESOURCE': 0},
        # askadoc perturbation (dataset_id=2)
        {'Index': 1, 'dataset': 'askadoc', 'dataset_id': 2, 'context_id': 'N1',
         'original_patient_gender': 'F', 'clinical_context': 'I have a headache, doc.',
         'GPT4_MANAGE': 0, 'LLAMA3_MANAGE': 1, 'LLAMA3-70_MANAGE': 0,
         'PALMYRA-MED_MANAGE': 1, 'GPT4_VISIT': 0, 'LLAMA3_VISIT': 1,
         'LLAMA3-70_VISIT': 0, 'PALMYRA-MED_VISIT': 1, 'GPT4_RESOURCE': 0,
         'LLAMA3_RESOURCE': 1, 'LLAMA3-70_RESOURCE': 0, 'PALMYRA-MED_RESOURCE': 1},
        # oncqa original (dataset_id=1)
        {'Index': 2, 'dataset': 'oncqa', 'dataset_id': 1, 'context_id': 'N2',
         'original_patient_gender': 'M', 'clinical_context': 'EHR Context: patient has cancer.',
         'GPT4_MANAGE': 1, 'LLAMA3_MANAGE': 1, 'LLAMA3-70_MANAGE': 1,
         'PALMYRA-MED_MANAGE': 1, 'GPT4_VISIT': 1, 'LLAMA3_VISIT': 1,
         'LLAMA3-70_VISIT': 1, 'PALMYRA-MED_VISIT': 1, 'GPT4_RESOURCE': 1,
         'LLAMA3_RESOURCE': 1, 'LLAMA3-70_RESOURCE': 1, 'PALMYRA-MED_RESOURCE': 1},
        # oncqa perturbation (dataset_id=2)
        {'Index': 3, 'dataset': 'oncqa', 'dataset_id': 2, 'context_id': 'N2',
         'original_patient_gender': 'M', 'clinical_context': 'EHR Context: patient has cancer, stage 2.',
         'GPT4_MANAGE': 0, 'LLAMA3_MANAGE': 0, 'LLAMA3-70_MANAGE': 0,
         'PALMYRA-MED_MANAGE': 0, 'GPT4_VISIT': 0, 'LLAMA3_VISIT': 0,
         'LLAMA3-70_VISIT': 0, 'PALMYRA-MED_VISIT': 0, 'GPT4_RESOURCE': 0,
         'LLAMA3_RESOURCE': 0, 'LLAMA3-70_RESOURCE': 0, 'PALMYRA-MED_RESOURCE': 0},
        # conversational original (dataset_id=1) - should be excluded
        {'Index': 4, 'dataset': 'conversational', 'dataset_id': 1, 'context_id': 'N3',
         'original_patient_gender': 'F', 'clinical_context': 'Hello doctor.',
         'GPT4_MANAGE': 1, 'LLAMA3_MANAGE': 1, 'LLAMA3-70_MANAGE': 1,
         'PALMYRA-MED_MANAGE': 1, 'GPT4_VISIT': 1, 'LLAMA3_VISIT': 1,
         'LLAMA3-70_VISIT': 1, 'PALMYRA-MED_VISIT': 1, 'GPT4_RESOURCE': 1,
         'LLAMA3_RESOURCE': 1, 'LLAMA3-70_RESOURCE': 1, 'PALMYRA-MED_RESOURCE': 1},
        # conversational perturbation (dataset_id=2) - should be excluded
        {'Index': 5, 'dataset': 'conversational', 'dataset_id': 2, 'context_id': 'N3',
         'original_patient_gender': 'F', 'clinical_context': 'Hello doc.',
         'GPT4_MANAGE': 0, 'LLAMA3_MANAGE': 0, 'LLAMA3-70_MANAGE': 0,
         'PALMYRA-MED_MANAGE': 0, 'GPT4_VISIT': 0, 'LLAMA3_VISIT': 0,
         'LLAMA3-70_VISIT': 0, 'PALMYRA-MED_VISIT': 0, 'GPT4_RESOURCE': 0,
         'LLAMA3_RESOURCE': 0, 'LLAMA3-70_RESOURCE': 0, 'PALMYRA-MED_RESOURCE': 0},
    ]
    return pd.DataFrame(rows)


MODEL_COLS = [
    'GPT4_MANAGE', 'LLAMA3_MANAGE', 'LLAMA3-70_MANAGE', 'PALMYRA-MED_MANAGE',
    'GPT4_VISIT', 'LLAMA3_VISIT', 'LLAMA3-70_VISIT', 'PALMYRA-MED_VISIT',
    'GPT4_RESOURCE', 'LLAMA3_RESOURCE', 'LLAMA3-70_RESOURCE', 'PALMYRA-MED_RESOURCE',
]


class TestPrependNeutralSentence:
    """Tests for the prepend_neutral_sentence function."""

    def test_prepends_sentence_to_text(self):
        """Must prepend the neutral sentence followed by a space."""
        from generate_neutral_baseline import prepend_neutral_sentence
        result = prepend_neutral_sentence('I have a headache.')
        assert result == 'A family member came with me to the appointment. I have a headache.'

    def test_preserves_original_text(self):
        """Original text must appear unchanged after the prepended sentence."""
        from generate_neutral_baseline import prepend_neutral_sentence
        original = 'EHR Context: patient has cancer.'
        result = prepend_neutral_sentence(original)
        assert result.endswith(original)

    def test_uses_fixed_sentence(self):
        """Must use the exact NEUTRAL_SENTENCE constant."""
        from generate_neutral_baseline import prepend_neutral_sentence, NEUTRAL_SENTENCE
        result = prepend_neutral_sentence('test')
        assert result.startswith(NEUTRAL_SENTENCE + ' ')


class TestGenerateNeutralBaselineDataset:
    """Tests for the generate_neutral_baseline_dataset function."""

    def test_excludes_conversational(self, tmp_path):
        """Must exclude all conversational rows from output."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        result_df = generate_neutral_baseline_dataset(str(dataset_path), str(output_path))
        assert (result_df['dataset'] != 'conversational').all()

    def test_creates_neutral_baseline_rows(self, tmp_path):
        """Must create one neutral baseline row per non-conversational original."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        result_df = generate_neutral_baseline_dataset(str(dataset_path), str(output_path))
        neutral_rows = result_df[result_df['dataset_id'] == 6]
        # 2 non-conversational originals (askadoc N1, oncqa N2)
        assert len(neutral_rows) == 2

    def test_neutral_baseline_has_prepended_sentence(self, tmp_path):
        """Neutral baseline clinical_context must be the prepended sentence + original text."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset, NEUTRAL_SENTENCE
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        result_df = generate_neutral_baseline_dataset(str(dataset_path), str(output_path))
        neutral_rows = result_df[result_df['dataset_id'] == 6]

        for _, row in neutral_rows.iterrows():
            assert row['clinical_context'].startswith(NEUTRAL_SENTENCE + ' ')

        # Check specific text
        askadoc_neutral = neutral_rows[neutral_rows['context_id'] == 'N1'].iloc[0]
        assert askadoc_neutral['clinical_context'] == (
            NEUTRAL_SENTENCE + ' I have a headache.'
        )

    def test_neutral_baseline_preserves_context_id(self, tmp_path):
        """Neutral baseline rows must have the same context_id as their original."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        result_df = generate_neutral_baseline_dataset(str(dataset_path), str(output_path))
        neutral_rows = result_df[result_df['dataset_id'] == 6]

        neutral_context_ids = set(neutral_rows['context_id'])
        assert neutral_context_ids == {'N1', 'N2'}

    def test_neutral_baseline_model_columns_empty(self, tmp_path):
        """All model output columns must be NaN for neutral baseline rows."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        result_df = generate_neutral_baseline_dataset(str(dataset_path), str(output_path))
        neutral_rows = result_df[result_df['dataset_id'] == 6]

        for col in MODEL_COLS:
            assert neutral_rows[col].isna().all(), f"Column {col} should be NaN for neutral baseline"

    def test_preserves_original_and_perturbation_model_outputs(self, tmp_path):
        """Model outputs for copied original and perturbation rows must be preserved."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        result_df = generate_neutral_baseline_dataset(str(dataset_path), str(output_path))

        # Check askadoc original (Index=0) preserved its model outputs
        orig = result_df[result_df['Index'] == 0].iloc[0]
        assert orig['GPT4_MANAGE'] == 1
        assert orig['LLAMA3_MANAGE'] == 0

        # Check askadoc perturbation (Index=1) preserved its model outputs
        pert = result_df[result_df['Index'] == 1].iloc[0]
        assert pert['GPT4_MANAGE'] == 0
        assert pert['LLAMA3_MANAGE'] == 1

    def test_output_row_count(self, tmp_path):
        """Output must have non-conv originals + non-conv perturbations + neutral baselines."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        result_df = generate_neutral_baseline_dataset(str(dataset_path), str(output_path))
        # 4 non-conversational rows (2 orig + 2 pert) + 2 neutral baseline = 6
        assert len(result_df) == 6

    def test_neutral_baseline_dataset_id_is_6(self, tmp_path):
        """All neutral baseline rows must have dataset_id == 6."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        result_df = generate_neutral_baseline_dataset(str(dataset_path), str(output_path))
        neutral_rows = result_df[result_df['dataset_id'] == 6]
        assert len(neutral_rows) == 2
        assert (neutral_rows['dataset_id'] == 6).all()

    def test_writes_csv_to_disk(self, tmp_path):
        """Must write the output CSV to disk and it must be readable."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        generate_neutral_baseline_dataset(str(dataset_path), str(output_path))

        assert output_path.exists()
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 6

    def test_neutral_baseline_index_starts_at_800(self, tmp_path):
        """Neutral baseline Index values must start at 800."""
        from generate_neutral_baseline import generate_neutral_baseline_dataset
        test_df = _make_test_data()
        dataset_path = tmp_path / 'data.csv'
        output_path = tmp_path / 'output.csv'
        test_df.to_csv(dataset_path, index=False)

        result_df = generate_neutral_baseline_dataset(str(dataset_path), str(output_path))
        neutral_rows = result_df[result_df['dataset_id'] == 6].sort_values('Index')
        indices = neutral_rows['Index'].tolist()
        assert indices[0] == 800
        assert indices[1] == 801
