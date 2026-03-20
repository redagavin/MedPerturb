# ABOUTME: Tests for loading real logits from MedPerturb experiment results
# ABOUTME: Validates clamping, majority vote, and condition extraction

import json
import math
import numpy as np
import pytest
import tempfile
import os


def _make_fake_results(n=5, p_yes_values=None):
    """Create minimal fake experiment JSON matching real format."""
    entries = []
    for i in range(n):
        entry = {"context_id": i}
        for q in ["MANAGE", "VISIT", "RESOURCE"]:
            p = p_yes_values[i] if p_yes_values else 0.5
            entry[f"original_{q}"] = {
                "logit_probs": p,
                "binary_answers": [1, 1, 0] if p > 0.5 else [0, 0, 1],
            }
        entries.append(entry)
    return entries


class TestLoadCondition:
    """Tests for load_condition extracting z_i and y_orig arrays."""

    def test_returns_z_and_y_orig(self):
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(5, [0.3, 0.5, 0.7, 0.2, 0.8]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            assert "z_i" in result
            assert "y_orig" in result
            assert len(result["z_i"]) == 5
            assert len(result["y_orig"]) == 5
        finally:
            os.unlink(path)

    def test_logit_conversion(self):
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(1, [0.7]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            expected = math.log(0.7 / 0.3)
            assert abs(result["z_i"][0] - expected) < 1e-10
        finally:
            os.unlink(path)

    def test_clamps_p_equals_1(self):
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(1, [1.0]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            expected = math.log((1.0 - 1e-6) / 1e-6)
            assert abs(result["z_i"][0] - expected) < 1e-6
            assert np.isfinite(result["z_i"][0])
        finally:
            os.unlink(path)

    def test_clamps_p_equals_0(self):
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(1, [0.0]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            expected = math.log(1e-6 / (1.0 - 1e-6))
            assert abs(result["z_i"][0] - expected) < 1e-6
            assert np.isfinite(result["z_i"][0])
        finally:
            os.unlink(path)

    def test_majority_vote_binary_answers(self):
        from load_experiment_logits import load_condition
        entries = [{"context_id": 0, "original_MANAGE": {
            "logit_probs": 0.6,
            "binary_answers": [1, 0, 1],
        }}]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(entries, f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            assert result["y_orig"][0] == 1
        finally:
            os.unlink(path)

    def test_different_questions(self):
        from load_experiment_logits import load_condition
        entries = [{
            "context_id": 0,
            "original_MANAGE": {"logit_probs": 0.3, "binary_answers": [0, 0, 0]},
            "original_VISIT": {"logit_probs": 0.8, "binary_answers": [1, 1, 1]},
            "original_RESOURCE": {"logit_probs": 0.5, "binary_answers": [0, 1, 0]},
        }]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(entries, f)
            path = f.name
        try:
            m = load_condition(path, "MANAGE")
            v = load_condition(path, "VISIT")
            r = load_condition(path, "RESOURCE")
            assert m["y_orig"][0] == 0
            assert v["y_orig"][0] == 1
            assert r["y_orig"][0] == 0
        finally:
            os.unlink(path)

    def test_z_i_is_numpy_float_array(self):
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(3, [0.3, 0.5, 0.7]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            assert isinstance(result["z_i"], np.ndarray)
            assert result["z_i"].dtype == np.float64
        finally:
            os.unlink(path)

    def test_y_orig_is_numpy_int_array(self):
        from load_experiment_logits import load_condition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(_make_fake_results(3, [0.3, 0.5, 0.7]), f)
            path = f.name
        try:
            result = load_condition(path, "MANAGE")
            assert isinstance(result["y_orig"], np.ndarray)
            assert set(np.unique(result["y_orig"])).issubset({0, 1})
        finally:
            os.unlink(path)
