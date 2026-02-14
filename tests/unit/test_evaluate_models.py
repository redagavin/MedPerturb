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
