# ABOUTME: Tests for evaluate_models.py extraction logic and chat template
# ABOUTME: Covers binary answer parsing fallback chain and source code inspection

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
