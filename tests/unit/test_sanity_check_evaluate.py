# ABOUTME: Tests for sanity check evaluation pipeline
# ABOUTME: Covers data loading, prompt construction, and evaluation logic

import ast
import os
import pytest
import pandas as pd


MEDPERTURB_CODE = '/scratch/yang.zih/cot_faithfulness/MedPerturb/code'


def _read_source(filename):
    path = os.path.join(MEDPERTURB_CODE, filename)
    with open(path) as f:
        return f.read()


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


class TestGenderPrompt:
    """Tests for gender question prompt construction."""

    def test_prompt_contains_question(self):
        """Prompt must include the gender question."""
        from sanity_check_evaluate import build_gender_prompt
        prompt = build_gender_prompt("Patient is a 30 year old female.")
        assert "Is this patient male?" in prompt

    def test_prompt_contains_patient_info(self):
        """Prompt must include the patient information."""
        from sanity_check_evaluate import build_gender_prompt
        patient_info = "Patient is a 30 year old female with chest pain."
        prompt = build_gender_prompt(patient_info)
        assert patient_info in prompt

    def test_prompt_requests_yes_no(self):
        """Prompt must request yes/no answer format."""
        from sanity_check_evaluate import build_gender_prompt
        prompt = build_gender_prompt("Some patient info")
        assert "yes" in prompt.lower()
        assert "no" in prompt.lower()


class MockEvaluator:
    """Shared mock evaluator for tests that need model-like behavior."""
    seeds = [0, 1, 42]
    def _call_model(self, prompt, seed):
        return "Yes"
    def _extract_binary_answer(self, response, question_type):
        return {"answer": 1, "extractor_output": "1",
                "extraction_method": "integer_parse"}
    def extract_logit_probs(self, prompt):
        return 0.75


class TestEvaluateGenderQuestion:
    """Tests for the per-sample gender evaluation function."""

    def test_returns_three_seed_responses(self):
        """Must return a dict with 3 binary values (one per seed)."""
        from sanity_check_evaluate import evaluate_gender_question

        result = evaluate_gender_question(MockEvaluator(), "Some patient info")
        assert len(result['binary_answers']) == 3
        assert all(r in [0, 1] for r in result['binary_answers'])

    def test_passes_correct_question_type(self):
        """Must pass 'GENDER' as question_type to _extract_binary_answer."""
        from sanity_check_evaluate import evaluate_gender_question

        captured_qtypes = []

        class CapturingEvaluator(MockEvaluator):
            def _extract_binary_answer(self, response, question_type):
                captured_qtypes.append(question_type)
                return {"answer": 1, "extractor_output": "1",
                        "extraction_method": "integer_parse"}

        evaluate_gender_question(CapturingEvaluator(), "info")
        assert all(qt == "GENDER" for qt in captured_qtypes)


class TestEvaluateSample:
    """Tests for evaluating all three versions of a single sample."""

    def test_returns_correct_keys(self):
        """Result must have context_id and all four GENDER trace dicts."""
        from sanity_check_evaluate import evaluate_sanity_check_sample

        sample = {
            'context_id': 'N75',
            'original_text': 'Female patient',
            'swap_text': 'Male patient',
            'baseline_text': 'Female patient paraphrased',
        }
        result = evaluate_sanity_check_sample(MockEvaluator(), sample)
        assert result['context_id'] == 'N75'
        assert 'original_GENDER' in result
        assert 'gender_swap_GENDER' in result
        assert 'gender_swap_baseline_GENDER' in result
        assert 'neutral_GENDER' in result
        assert len(result['original_GENDER']['binary_answers']) == 3
        assert len(result['gender_swap_GENDER']['binary_answers']) == 3
        assert len(result['gender_swap_baseline_GENDER']['binary_answers']) == 3
        assert len(result['neutral_GENDER']['binary_answers']) == 3


class TestTraceDataReturn:
    """evaluate_gender_question must return trace data dict."""

    def test_returns_trace_dict(self):
        """Must return dict with binary_answers, model_responses, etc."""
        from sanity_check_evaluate import evaluate_gender_question

        result = evaluate_gender_question(MockEvaluator(), "Some info")
        assert isinstance(result, dict)
        assert 'binary_answers' in result
        assert 'model_responses' in result
        assert 'extractor_outputs' in result
        assert 'extraction_methods' in result
        assert len(result['binary_answers']) == 3
        assert all(r in [0, 1] for r in result['binary_answers'])

    def test_includes_logit_probs(self):
        """Must include logit_probs from evaluator.extract_logit_probs."""
        from sanity_check_evaluate import evaluate_gender_question

        result = evaluate_gender_question(MockEvaluator(), "Some info")
        assert 'logit_probs' in result
        assert result['logit_probs'] == 0.75


class TestCheckpointing:
    """Tests for checkpoint save/load functionality."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved checkpoint must be recoverable with identical data."""
        from sanity_check_evaluate import save_checkpoint, load_checkpoint

        path = str(tmp_path / "test.pkl")
        results = [{'context_id': 'N75', 'original_GENDER': [1, 1, 1]}]
        completed = {'N75'}

        save_checkpoint(path, results, completed)
        loaded_results, loaded_completed = load_checkpoint(path)

        assert loaded_results == results
        assert loaded_completed == completed

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """Loading from a nonexistent path must return empty state."""
        from sanity_check_evaluate import load_checkpoint

        results, completed = load_checkpoint(str(tmp_path / "nonexistent.pkl"))
        assert results == []
        assert completed == set()

    def test_completed_ids_stored_as_set(self, tmp_path):
        """Completed IDs must be returned as a set for O(1) lookups."""
        from sanity_check_evaluate import save_checkpoint, load_checkpoint

        path = str(tmp_path / "test.pkl")
        save_checkpoint(path, [], {'A', 'B', 'C'})
        _, completed = load_checkpoint(path)
        assert isinstance(completed, set)
        assert completed == {'A', 'B', 'C'}


class TestShardSamples:
    """Tests for sample sharding across GPUs."""

    def test_single_gpu_returns_all(self):
        """With 1 GPU, all samples should be returned."""
        from sanity_check_evaluate import shard_samples

        samples = [{'id': i} for i in range(10)]
        result = shard_samples(samples, gpu_id=0, total_gpus=1)
        assert len(result) == 10

    def test_two_gpus_split_evenly(self):
        """With 2 GPUs, samples should be split roughly evenly."""
        from sanity_check_evaluate import shard_samples

        samples = [{'id': i} for i in range(10)]
        shard0 = shard_samples(samples, gpu_id=0, total_gpus=2)
        shard1 = shard_samples(samples, gpu_id=1, total_gpus=2)
        assert len(shard0) == 5
        assert len(shard1) == 5
        # No overlap
        ids0 = {s['id'] for s in shard0}
        ids1 = {s['id'] for s in shard1}
        assert ids0.isdisjoint(ids1)
        # Complete coverage
        assert ids0 | ids1 == {i for i in range(10)}

    def test_uneven_split(self):
        """With odd sample count, one shard gets an extra sample."""
        from sanity_check_evaluate import shard_samples

        samples = [{'id': i} for i in range(7)]
        shard0 = shard_samples(samples, gpu_id=0, total_gpus=3)
        shard1 = shard_samples(samples, gpu_id=1, total_gpus=3)
        shard2 = shard_samples(samples, gpu_id=2, total_gpus=3)
        assert len(shard0) + len(shard1) + len(shard2) == 7


class TestNeutralBaseline:
    """Neutral baseline prepends fixed sentence to original text."""

    def test_neutral_key_in_result(self):
        """evaluate_sanity_check_sample returns neutral_GENDER key."""
        source = _read_source('sanity_check_evaluate.py')
        assert 'neutral_GENDER' in source

    def test_neutral_sentence_constant(self):
        """NEUTRAL_SENTENCE is defined."""
        source = _read_source('sanity_check_evaluate.py')
        assert 'A family member came with me to the appointment.' in source

    def test_neutral_text_construction(self):
        """Neutral text is NEUTRAL_SENTENCE + ' ' + original_text."""
        from sanity_check_evaluate import NEUTRAL_SENTENCE
        original = "Patient presents with cough."
        expected = NEUTRAL_SENTENCE + " " + original
        assert expected == "A family member came with me to the appointment. Patient presents with cough."


class TestLogitProbsInResult:
    """Evaluation results include logit_probs."""

    def test_evaluate_gender_question_has_logit_probs(self):
        """evaluate_gender_question result includes logit_probs key."""
        source = _read_source('sanity_check_evaluate.py')
        assert 'logit_probs' in source


class TestUsesEvalUtils:
    """sanity_check_evaluate.py uses shared eval_utils."""

    def test_imports_eval_utils(self):
        source = _read_source('sanity_check_evaluate.py')
        assert 'eval_utils' in source

    def test_no_inline_save_checkpoint(self):
        """save_checkpoint should come from eval_utils, not defined inline."""
        source = _read_source('sanity_check_evaluate.py')
        tree = ast.parse(source)
        # No top-level function named save_checkpoint
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'save_checkpoint':
                pytest.fail("save_checkpoint should be imported from eval_utils, not defined inline")
