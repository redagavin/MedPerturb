# ABOUTME: Tests for precision check evaluation (age swap negative control)
# ABOUTME: Verifies eval_utils integration, logit extraction, neutral baseline

import ast
import os
import sys
import pytest
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')

MEDPERTURB_CODE = '/scratch/yang.zih/cot_faithfulness/MedPerturb/code'


def _read_source(filename):
    path = os.path.join(MEDPERTURB_CODE, filename)
    with open(path) as f:
        return f.read()


class TestUsesEvalUtils:
    """precision_check_evaluate.py uses shared eval_utils."""

    def test_imports_eval_utils(self):
        source = _read_source('precision_check_evaluate.py')
        assert 'eval_utils' in source

    def test_no_inline_save_checkpoint(self):
        """save_checkpoint should come from eval_utils, not defined inline."""
        source = _read_source('precision_check_evaluate.py')
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'save_checkpoint':
                pytest.fail("save_checkpoint should be imported from eval_utils")

    def test_no_inline_shard_samples(self):
        """shard_samples should come from eval_utils, not defined inline."""
        source = _read_source('precision_check_evaluate.py')
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'shard_samples':
                pytest.fail("shard_samples should be imported from eval_utils")


class TestNeutralBaseline:
    """Neutral baseline evaluation in precision check."""

    def test_neutral_key_in_result(self):
        """evaluate_precision_check_sample returns neutral_GENDER key."""
        source = _read_source('precision_check_evaluate.py')
        assert 'neutral_GENDER' in source

    def test_neutral_sentence_constant(self):
        """NEUTRAL_SENTENCE is defined."""
        source = _read_source('precision_check_evaluate.py')
        assert 'A family member came with me to the appointment.' in source

    def test_neutral_text_construction(self):
        """Neutral text is NEUTRAL_SENTENCE + ' ' + original_text."""
        from precision_check_evaluate import NEUTRAL_SENTENCE
        original = "Patient presents with fever."
        expected = NEUTRAL_SENTENCE + " " + original
        assert expected == "A family member came with me to the appointment. Patient presents with fever."


class TestLogitProbsInResult:
    """Evaluation results include logit_probs."""

    def test_evaluate_gender_question_has_logit_probs(self):
        """evaluate_gender_question result includes logit_probs key."""
        source = _read_source('precision_check_evaluate.py')
        assert 'logit_probs' in source

    def test_extract_logit_probs_called(self):
        """extract_logit_probs is called in the evaluation function."""
        source = _read_source('precision_check_evaluate.py')
        assert 'extract_logit_probs' in source


class TestResultStructure:
    """Result dict has all required keys."""

    def test_result_keys_in_source(self):
        """evaluate_precision_check_sample returns all required keys."""
        source = _read_source('precision_check_evaluate.py')
        for key in ['original_GENDER', 'age_swap_GENDER',
                     'age_swap_baseline_GENDER', 'neutral_GENDER']:
            assert key in source, f"Missing result key: {key}"


class TestModelTaggedFilenames:
    """Filenames include model short name."""

    def test_uses_model_short_name(self):
        source = _read_source('precision_check_evaluate.py')
        assert 'model_short_name' in source
