# ABOUTME: GPU integration tests for model inference correctness
# ABOUTME: Verifies token-level slicing, binary extraction, and perturbation generation

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))


@pytest.mark.gpu
class TestModelInference:
    """Integration tests that require GPU. Run via srun on 177huntington."""

    @pytest.fixture(scope="class")
    def evaluator(self):
        """Load ModelEvaluator once for all tests in this class."""
        from evaluate_models import ModelEvaluator
        return ModelEvaluator("meta-llama/Llama-3.1-8B-Instruct")

    def test_response_does_not_contain_prompt(self, evaluator):
        """Token-level slicing must produce response without prompt text."""
        prompt = "Answer yes or no: Is the sky blue?\nAnswer:"
        response = evaluator._call_model(prompt, seed=42)
        assert not response.startswith("Answer yes or no"), (
            f"Response contains prompt text: {response[:100]}"
        )

    def test_binary_extraction_returns_valid(self, evaluator):
        """Binary answer extractor must return trace dict with valid binary answer."""
        result = evaluator._extract_binary_answer(
            "Yes, the patient should visit.", "VISIT"
        )
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result["answer"] in [0, 1], f"Expected 0 or 1, got {result['answer']}"
        assert "extractor_output" in result
        assert "extraction_method" in result

    def test_evaluate_triage_returns_correct_structure(self, evaluator):
        """evaluate_triage must return dict with MANAGE/VISIT/RESOURCE trace data."""
        sample_context = "A 45-year-old woman presents with mild headache for 2 days."
        results = evaluator.evaluate_triage(sample_context)
        for key in ["MANAGE", "VISIT", "RESOURCE"]:
            assert key in results, f"Missing key: {key}"
            trace = results[key]
            assert len(trace["binary_answers"]) == 3, (
                f"{key} should have 3 binary_answers (3 seeds), got {len(trace['binary_answers'])}"
            )
            for val in trace["binary_answers"]:
                assert val in [0, 1], f"{key} response {val} is not binary"
            assert len(trace["model_responses"]) == 3
            assert len(trace["extractor_outputs"]) == 3
            assert len(trace["extraction_methods"]) == 3

    def test_long_prompt_does_not_crash(self, evaluator):
        """max_new_tokens must handle prompts longer than 512 tokens without crashing."""
        long_context = "The patient reports: " + " ".join(["symptom"] * 300)
        prompt = f"Patient information:\n{long_context}\nAnswer (yes/no):"
        response = evaluator._call_model(prompt, seed=42)
        assert isinstance(response, str)


@pytest.mark.gpu
class TestPerturbationInference:
    """Integration tests for perturbation generation."""

    @pytest.fixture(scope="class")
    def perturber(self):
        from perturb_data import ClinicalContextPerturber
        return ClinicalContextPerturber(
            model_name="meta-llama/Llama-3.1-8B-Instruct"
        )

    def test_gender_swap_produces_output(self, perturber):
        """Gender swap perturbation must produce non-empty output."""
        text = "A 45-year-old woman presents with chest pain."
        result = perturber.perturb_gender(text, "swap")
        assert len(result) > 0, "Gender swap produced empty output"
        assert "Please swap the gender" not in result, "Output contains prompt text"

    def test_stylistic_produces_output(self, perturber):
        """Stylistic perturbation must produce non-empty output."""
        text = "A 45-year-old woman presents with chest pain."
        result = perturber.perturb_stylistic(text, "uncertain")
        assert len(result) > 0, "Stylistic perturbation produced empty output"
        assert "Add uncertain language" not in result, "Output contains prompt text"
