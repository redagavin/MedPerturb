# ABOUTME: Source code inspection tests verifying correct patterns in pipeline files
# ABOUTME: No GPU or model loading required — inspects source text and AST only

import ast
import os
import sys
import pytest

MEDPERTURB_CODE = '/scratch/yang.zih/cot_faithfulness/MedPerturb/code'
sys.path.insert(0, MEDPERTURB_CODE)


def _read_source(filename):
    """Read source code from the code directory."""
    path = os.path.join(MEDPERTURB_CODE, filename)
    with open(path) as f:
        return f.read()


def _find_keyword_in_generate_calls(source, keyword_name):
    """Find instances of a keyword argument in generate() calls in source code."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if this is a .generate() call
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'generate':
                for kw in node.keywords:
                    if kw.arg == keyword_name:
                        found.append(kw.lineno)
    return found


class TestMaxNewTokens:
    """All HuggingFace generate() calls must use max_new_tokens, not max_length."""

    def test_perturb_data_uses_max_new_tokens(self):
        """perturb_data.py must use max_new_tokens, not max_length, in generate() calls."""
        source = _read_source('perturb_data.py')
        bad_lines = _find_keyword_in_generate_calls(source, 'max_length')
        assert bad_lines == [], (
            f"Found max_length in generate() at line(s) {bad_lines}. "
            "Should be max_new_tokens."
        )

    def test_evaluate_models_uses_max_new_tokens(self):
        """evaluate_models.py must use max_new_tokens in all generate() calls."""
        source = _read_source('evaluate_models.py')
        bad_lines = _find_keyword_in_generate_calls(source, 'max_length')
        assert bad_lines == [], (
            f"Found max_length in generate() at line(s) {bad_lines}. "
            "Should be max_new_tokens."
        )


class TestTokenLevelSlicing:
    """Response extraction must use token-level slicing, not string-level."""

    def test_evaluate_models_no_string_slicing(self):
        """evaluate_models.py must not use [len(prompt):] for response extraction."""
        source = _read_source('evaluate_models.py')
        assert '[len(prompt):]' not in source, (
            "String-level slicing found. Must use token-level: "
            "outputs[0][inputs['input_ids'].shape[1]:]"
        )

    def test_perturb_data_no_string_slicing(self):
        """perturb_data.py must not use [len(full_prompt):] for response extraction."""
        source = _read_source('perturb_data.py')
        assert '[len(full_prompt):]' not in source, (
            "String-level slicing found. Must use token-level slicing."
        )


class TestExtractorModelDevice:
    """Extractor model should use device_map='auto' to avoid OOM with 70B main models."""

    def test_extractor_model_not_using_to_device(self):
        """Extractor model must use device_map='auto', not .to(self.device)."""
        source = _read_source('evaluate_models.py')
        tree = ast.parse(source)

        # Find assignments to self.extractor_model
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Attribute) and
                            target.attr == 'extractor_model'):
                        # Check if the value is a .to() call (bad pattern)
                        if (isinstance(node.value, ast.Call) and
                                isinstance(node.value.func, ast.Attribute) and
                                node.value.func.attr == 'to'):
                            pytest.fail(
                                f"Line {node.lineno}: extractor_model uses .to(). "
                                "Should use device_map='auto' in from_pretrained()."
                            )


class TestPerturbationTypesConsistency:
    """PERTURBATION_TYPES must be identical across pipeline files."""

    def test_types_match_across_files(self):
        """generate_baselines.py and batch_evaluate.py must have same PERTURBATION_TYPES."""
        from generate_baselines import PERTURBATION_TYPES as baseline_types
        from batch_evaluate import PERTURBATION_TYPES as eval_types
        assert baseline_types == eval_types


class TestPipelineUsesIndexKey:
    """Pipeline files must use Index (unique row ID) as primary key."""

    def test_generate_baselines_uses_index(self):
        """generate_baselines.py must reference row['Index']."""
        source = _read_source('generate_baselines.py')
        assert "row['Index']" in source, (
            "generate_baselines.py must key by row['Index']"
        )

    def test_batch_evaluate_uses_index(self):
        """batch_evaluate.py must reference row['Index'] or df['Index']."""
        source = _read_source('batch_evaluate.py')
        assert "'Index'" in source, (
            "batch_evaluate.py must use 'Index' as primary key"
        )
