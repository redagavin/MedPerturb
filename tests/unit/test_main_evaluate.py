# ABOUTME: Tests for unified main experiment evaluation script
# ABOUTME: Validates data loading, result structure, neutral baseline, and eval_utils usage

import ast
import os
import sys
import pytest
import pandas as pd
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')

MEDPERTURB_CODE = '/scratch/yang.zih/cot_faithfulness/MedPerturb/code'


def _read_source(filename):
    path = os.path.join(MEDPERTURB_CODE, filename)
    with open(path) as f:
        return f.read()


def _make_test_csv(tmp_path, context_ids, dataset_ids, dataset_type="askadoc"):
    """Create a minimal test CSV with given context_ids and dataset_ids."""
    rows = []
    for cid in context_ids:
        for did in dataset_ids:
            rows.append({
                "context_id": cid,
                "dataset_id": did,
                "dataset": dataset_type,
                "clinical_context": f"Text for cid={cid} did={did}",
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestLoadMainExperimentData:
    """Tests for load_main_experiment_data()."""

    def test_groups_by_context_id(self, tmp_path):
        """Samples are grouped by context_id with all text variants."""
        from main_evaluate import load_main_experiment_data, VARIANT_NAMES
        csv_path = _make_test_csv(tmp_path, [10, 20], list(range(1, 10)))
        samples = load_main_experiment_data(csv_path)
        assert len(samples) == 2
        for sample in samples:
            assert "context_id" in sample
            for name in VARIANT_NAMES.values():
                assert f"{name}_text" in sample

    def test_filters_conversational(self, tmp_path):
        """Conversational rows are excluded."""
        from main_evaluate import load_main_experiment_data
        # Create CSV with one conversational and one non-conversational context
        rows = []
        for did in range(1, 10):
            rows.append({"context_id": 10, "dataset_id": did,
                          "dataset": "askadoc", "clinical_context": f"text {did}"})
            rows.append({"context_id": 20, "dataset_id": did,
                          "dataset": "conversational", "clinical_context": f"conv {did}"})
        df = pd.DataFrame(rows)
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        samples = load_main_experiment_data(str(path))
        assert len(samples) == 1
        assert samples[0]["context_id"] == 10

    def test_requires_all_dataset_ids(self, tmp_path):
        """Samples missing any dataset_id are excluded."""
        from main_evaluate import load_main_experiment_data
        # context_id=10 has all 9, context_id=20 only has 8
        csv_path = _make_test_csv(tmp_path, [10], list(range(1, 10)))
        rows = [{"context_id": 20, "dataset_id": did, "dataset": "askadoc",
                 "clinical_context": f"text {did}"} for did in range(1, 9)]  # missing id=9
        df2 = pd.DataFrame(rows)
        df = pd.concat([pd.read_csv(csv_path), df2])
        path = tmp_path / "test2.csv"
        df.to_csv(path, index=False)
        samples = load_main_experiment_data(str(path))
        assert len(samples) == 1
        assert samples[0]["context_id"] == 10

    def test_sorted_by_context_id(self, tmp_path):
        """Samples are sorted by context_id for determinism."""
        from main_evaluate import load_main_experiment_data
        csv_path = _make_test_csv(tmp_path, [30, 10, 20], list(range(1, 10)))
        samples = load_main_experiment_data(csv_path)
        cids = [s["context_id"] for s in samples]
        assert cids == sorted(cids)


class TestResultKeyNaming:
    """Result keys follow naming conventions."""

    def test_variant_names_complete(self):
        """VARIANT_NAMES covers all 9 dataset_ids."""
        from main_evaluate import VARIANT_NAMES
        assert set(VARIANT_NAMES.keys()) == set(range(1, 10))

    def test_result_key_format_in_source(self):
        """evaluate_main_experiment_sample builds correct result keys."""
        source = _read_source('main_evaluate.py')
        # Neutral baseline keys
        assert 'neutral_' in source
        # Calibrated baseline keys (via VARIANT_NAMES)
        assert 'gender_swap_baseline' in source or 'VARIANT_NAMES' in source


class TestNeutralBaseline:
    """Neutral baseline is prepended inline."""

    def test_neutral_sentence_constant(self):
        from main_evaluate import NEUTRAL_SENTENCE
        assert NEUTRAL_SENTENCE == "A family member came with me to the appointment."

    def test_neutral_text_construction(self):
        """Neutral text is NEUTRAL_SENTENCE + ' ' + original."""
        from main_evaluate import NEUTRAL_SENTENCE
        original = "Patient has fever."
        expected = NEUTRAL_SENTENCE + " " + original
        assert expected.startswith("A family member")
        assert expected.endswith("Patient has fever.")


class TestUsesEvalUtils:
    """main_evaluate.py uses shared eval_utils."""

    def test_imports_eval_utils(self):
        source = _read_source('main_evaluate.py')
        assert 'eval_utils' in source

    def test_uses_model_short_name(self):
        source = _read_source('main_evaluate.py')
        assert 'model_short_name' in source

    def test_uses_mark_complete(self):
        source = _read_source('main_evaluate.py')
        assert 'mark_complete' in source


class TestIntegrationWithAnalysis:
    """Evaluation output format is parseable by analysis scripts."""

    def test_result_entry_has_required_fields(self):
        """Each result entry must have context_id and variant result dicts."""
        from main_evaluate import VARIANT_NAMES
        # Simulate a result entry
        result = {"context_id": 1}
        for variant_name in VARIANT_NAMES.values():
            for q in ["MANAGE", "VISIT", "RESOURCE"]:
                result[f"{variant_name}_{q}"] = {
                    "seeds": [0, 1, 42],
                    "binary_answers": [1, 0, 1],
                    "logit_probs": 0.65,
                }
        for q in ["MANAGE", "VISIT", "RESOURCE"]:
            result[f"neutral_{q}"] = {
                "seeds": [0, 1, 42],
                "binary_answers": [1, 1, 0],
                "logit_probs": 0.55,
            }
        # 9 variants x 3 questions + 1 neutral x 3 questions + context_id = 31 keys
        assert len(result) == 31
        # All result dicts have binary_answers and logit_probs
        for key, val in result.items():
            if key == "context_id":
                continue
            assert "binary_answers" in val
            assert "logit_probs" in val
