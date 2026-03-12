# ABOUTME: Tests for evaluate_models.py extraction logic and chat template
# ABOUTME: Covers binary answer parsing fallback chain and source code inspection

import ast
import pytest
import sys
import os

MEDPERTURB_CODE = '/scratch/yang.zih/cot_faithfulness/MedPerturb/code'
sys.path.insert(0, MEDPERTURB_CODE)


def _read_source(filename):
    """Read source code from the code directory."""
    path = os.path.join(MEDPERTURB_CODE, filename)
    with open(path) as f:
        return f.read()


class TestParseBinaryAnswer:
    """Tests for the four-layer binary answer extraction chain."""

    # Layer 1: Integer parse
    @pytest.mark.parametrize("extractor_output,expected", [
        ("0", 0),
        ("1", 1),
        ("  1  ", 1),
        ("  0\n", 0),
    ])
    def test_layer1_integer_parse(self, extractor_output, expected):
        """Clean integer strings parsed directly."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer(extractor_output, "any response")
        assert answer == expected
        assert method == "integer_parse"

    @pytest.mark.parametrize("extractor_output", ["2", "-1", "10", "abc"])
    def test_layer1_rejects_non_binary(self, extractor_output):
        """Non-binary integers and non-integers fall through to later layers."""
        from evaluate_models import parse_binary_answer
        _, method = parse_binary_answer(extractor_output, "Yes.")
        assert method != "integer_parse"

    # Layer 2: Word-boundary regex on extractor output
    @pytest.mark.parametrize("extractor_output,expected", [
        ("yes", 1),
        ("Yes", 1),
        ("YES", 1),
        ("The answer is yes.", 1),
        ("no", 0),
        ("No", 0),
        ("The answer is no.", 0),
        ("The answer is 1.", 1),
        ("The answer is 0.", 0),
    ])
    def test_layer2_text_match(self, extractor_output, expected):
        """Word-boundary regex catches yes/no/1/0 in extractor output."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer(extractor_output, "unrelated")
        assert answer == expected
        assert method == "extractor_text_match"

    @pytest.mark.parametrize("extractor_output", [
        "yesterday",
        "noble",
        "knowledge",
        "announce",
    ])
    def test_layer2_no_false_positive(self, extractor_output):
        """Substring matches must NOT trigger — word boundary prevents 'yesterday' matching 'yes'."""
        from evaluate_models import parse_binary_answer
        _, method = parse_binary_answer(extractor_output, "Yes.")
        assert method != "extractor_text_match"

    # Layer 3: Word-boundary regex on model response
    def test_layer3_model_says_yes(self):
        """When extractor fails, regex on model response 'Yes.' -> 1."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("unclear xyz", "Yes.")
        assert answer == 1
        assert method == "model_response_regex"

    def test_layer3_model_says_no(self):
        """When extractor fails, regex on model response 'No, I don't recommend.' -> 0."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("unclear xyz", "No, I don't recommend.")
        assert answer == 0
        assert method == "model_response_regex"

    # Layer 4: Default
    def test_layer4_empty_inputs(self):
        """Empty strings -> default 0."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("", "")
        assert answer == 0
        assert method == "default"

    def test_layer4_no_match_anywhere(self):
        """Gibberish in both inputs -> default 0."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("xyzzy", "abcdef gibberish")
        assert answer == 0
        assert method == "default"

    # Priority: Layer 1 > Layer 2 > Layer 3 > Layer 4
    def test_layer1_takes_priority_over_layer2(self):
        """'1' matches both integer parse and text match — integer parse wins."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("1", "any")
        assert method == "integer_parse"

    def test_layer2_takes_priority_over_layer3(self):
        """Extractor 'yes' matches layer 2 even if model response also has 'yes'."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("yes", "Yes.")
        assert method == "extractor_text_match"

    def test_layer2_both_yes_and_no_returns_first_match(self):
        """When extractor output contains both 'yes' and 'no', 'yes' wins (checked first).

        This documents a positional bias: the yes/1 regex is checked before no/0.
        In practice this is rare since the extractor is asked to produce '0' or '1'.
        """
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("No, wait, yes", "unrelated")
        assert answer == 1
        assert method == "extractor_text_match"

    def test_layer2_both_no_and_yes_still_returns_yes(self):
        """Even when 'no' appears first textually, the regex check order determines result."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("no yes", "unrelated")
        assert answer == 1
        assert method == "extractor_text_match"

    # Layer 2a: Explicit conclusion pattern overrides generic search
    def test_conclusion_pattern_overrides_generic_yes(self):
        """When extractor mentions 'yes' in explanation but concludes 'binary answer is 0',
        the explicit conclusion must win."""
        from evaluate_models import parse_binary_answer
        extractor = (
            "The response does not explicitly state 'yes' or 'no' to the question. "
            "However, it can be inferred that the response is not a direct 'yes'. "
            "Therefore, the binary answer is 0."
        )
        answer, method = parse_binary_answer(extractor, "I cannot provide medical advice.")
        assert answer == 0
        assert method == "extractor_text_match"

    def test_conclusion_pattern_finds_explicit_1(self):
        """Explicit 'binary answer is: 1' is correctly extracted."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("The binary answer is: 1", "Yes")
        assert answer == 1
        assert method == "extractor_text_match"

    def test_multiple_conclusions_last_wins(self):
        """When extractor analyzes multiple sub-questions, the last conclusion is used."""
        from evaluate_models import parse_binary_answer
        extractor = (
            "A. Yes - the binary answer is 1.\n"
            "B. Yes - the binary answer is 1.\n"
            "C. No - the binary answer is 0."
        )
        answer, method = parse_binary_answer(extractor, "A. Yes B. Yes C. No")
        assert answer == 0
        assert method == "extractor_text_match"

    def test_generic_search_when_no_conclusion_pattern(self):
        """When no 'binary answer' pattern exists, generic yes/no search still works."""
        from evaluate_models import parse_binary_answer
        answer, method = parse_binary_answer("The answer is yes.", "unrelated")
        assert answer == 1
        assert method == "extractor_text_match"


class TestChatTemplate:
    """Chat template must be applied before model/extractor inference."""

    def test_call_model_applies_chat_template(self):
        """_call_model must use apply_chat_template for HuggingFace models."""
        source = _read_source('evaluate_models.py')
        in_method = False
        found = False
        for line in source.split('\n'):
            if 'def _call_model(' in line:
                in_method = True
            elif in_method and line.strip().startswith('def '):
                break
            elif in_method and 'apply_chat_template' in line:
                found = True
                break
        assert found, "_call_model must use apply_chat_template"

    def test_extractor_applies_chat_template(self):
        """_extract_binary_answer must use apply_chat_template for extractor."""
        source = _read_source('evaluate_models.py')
        in_method = False
        found = False
        for line in source.split('\n'):
            if 'def _extract_binary_answer(' in line:
                in_method = True
            elif in_method and line.strip().startswith('def '):
                break
            elif in_method and 'apply_chat_template' in line:
                found = True
                break
        assert found, "_extract_binary_answer must use apply_chat_template"


class TestExtractBinaryAnswerReturn:
    """_extract_binary_answer must return trace dict, not bare int."""

    def test_return_type_is_dict_in_source(self):
        """_extract_binary_answer must return a dict with answer, extractor_output, extraction_method."""
        source = _read_source('evaluate_models.py')
        in_method = False
        found_answer = False
        found_extractor_output = False
        found_extraction_method = False
        for line in source.split('\n'):
            if 'def _extract_binary_answer(' in line:
                in_method = True
            elif in_method and line.strip().startswith('def '):
                break
            elif in_method:
                if '"answer"' in line or "'answer'" in line:
                    found_answer = True
                if '"extractor_output"' in line or "'extractor_output'" in line:
                    found_extractor_output = True
                if '"extraction_method"' in line or "'extraction_method'" in line:
                    found_extraction_method = True
        assert found_answer, "_extract_binary_answer must return 'answer' in dict"
        assert found_extractor_output, "_extract_binary_answer must return 'extractor_output' in dict"
        assert found_extraction_method, "_extract_binary_answer must return 'extraction_method' in dict"

    def test_uses_parse_binary_answer(self):
        """_extract_binary_answer must delegate to parse_binary_answer."""
        source = _read_source('evaluate_models.py')
        in_method = False
        found = False
        for line in source.split('\n'):
            if 'def _extract_binary_answer(' in line:
                in_method = True
            elif in_method and line.strip().startswith('def '):
                break
            elif in_method and 'parse_binary_answer' in line:
                found = True
                break
        assert found, "_extract_binary_answer must call parse_binary_answer"


class TestEvaluateTriageTraceData:
    """evaluate_triage must return trace data alongside binary answers."""

    def test_trace_fields_present_in_source(self):
        """evaluate_triage must include all trace data fields."""
        source = _read_source('evaluate_models.py')
        in_method = False
        required_fields = ['model_responses', 'extractor_outputs', 'extraction_methods',
                           'binary_answers', 'seeds']
        found_fields = set()
        for line in source.split('\n'):
            if 'def evaluate_triage(' in line:
                in_method = True
            elif in_method and (line.strip().startswith('def ') and 'evaluate_triage' not in line):
                break
            elif in_method:
                for field in required_fields:
                    if f'"{field}"' in line or f"'{field}'" in line:
                        found_fields.add(field)
        missing = set(required_fields) - found_fields
        assert not missing, f"evaluate_triage missing trace fields: {missing}"


class TestLogitExtraction:
    """Tests for Yes/No token ID validation and logit extraction."""

    def test_validate_yes_no_tokens_exist(self):
        """ModelEvaluator has yes_token_id and no_token_id after init."""
        source = _read_source('evaluate_models.py')
        assert '_validate_yes_no_tokens' in source

    def test_validate_yes_no_tokens_called_in_init(self):
        """_validate_yes_no_tokens is called during __init__."""
        source = _read_source('evaluate_models.py')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                init_source = ast.get_source_segment(source, node)
                assert '_validate_yes_no_tokens' in init_source

    def test_extract_logit_probs_method_exists(self):
        """ModelEvaluator must have extract_logit_probs method."""
        source = _read_source('evaluate_models.py')
        tree = ast.parse(source)
        method_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'extract_logit_probs':
                method_names.append(node.name)
        assert 'extract_logit_probs' in method_names

    def test_extract_logit_probs_uses_no_grad(self):
        """extract_logit_probs must use torch.no_grad() context."""
        source = _read_source('evaluate_models.py')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'extract_logit_probs':
                func_source = ast.get_source_segment(source, node)
                assert 'no_grad' in func_source, \
                    "extract_logit_probs must use torch.no_grad()"

    def test_extract_logit_probs_does_not_call_generate(self):
        """extract_logit_probs must NOT call model.generate()."""
        source = _read_source('evaluate_models.py')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'extract_logit_probs':
                func_source = ast.get_source_segment(source, node)
                assert '.generate(' not in func_source, \
                    "extract_logit_probs must use model(**inputs), not model.generate()"

    def test_extract_logit_probs_guards_openai(self):
        """extract_logit_probs must raise NotImplementedError for OpenAI models."""
        source = _read_source('evaluate_models.py')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'extract_logit_probs':
                func_source = ast.get_source_segment(source, node)
                assert 'NotImplementedError' in func_source, \
                    "extract_logit_probs must guard against OpenAI model_type"

    def test_evaluate_triage_returns_logit_probs(self):
        """evaluate_triage result must include logit_probs key."""
        source = _read_source('evaluate_models.py')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_triage':
                func_source = ast.get_source_segment(source, node)
                assert 'logit_probs' in func_source, \
                    "evaluate_triage must include logit_probs in result dict"
