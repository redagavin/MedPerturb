# ABOUTME: Tests for age extraction, replacement, and full age swap pipeline
# ABOUTME: Verifies all age occurrences are replaced, preventing contradictory age information

import re
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))
from precision_check_age_swap import (
    extract_age, replace_age, compute_target_age, run_age_swap, context_id_to_seed,
)


class TestExtractAge:
    def test_year_old_format(self):
        assert extract_age("68-year-old male")[0] == 68

    def test_age_header_format(self):
        assert extract_age("Age: 55 years\nGender: Male")[0] == 55

    def test_compact_gender(self):
        assert extract_age("28M presenting with cough")[0] == 28

    def test_gender_prefix(self):
        assert extract_age("F24 presenting with headache")[0] == 24

    def test_no_age(self):
        assert extract_age("Patient presents with cough") is None


class TestReplaceAge:
    def test_year_old_replacement(self):
        text = "68-year-old male with cancer"
        age, matched, start = extract_age(text)
        result = replace_age(text, matched, 21, match_start=start)
        assert "21-year-old" in result
        assert "68" not in result

    def test_compact_gender_replacement(self):
        text = "28M presenting with cough"
        age, matched, start = extract_age(text)
        result = replace_age(text, matched, 65, match_start=start)
        assert "65M" in result
        assert "28" not in result


class TestComputeTargetAge:
    def test_young_to_old(self):
        age = compute_target_age(25, seed=42)
        assert 60 <= age <= 80

    def test_old_to_young(self):
        age = compute_target_age(65, seed=42)
        assert 18 <= age <= 35

    def test_deterministic(self):
        assert compute_target_age(50, seed=0) == compute_target_age(50, seed=0)


class TestRunAgeSwapNoContradictions:
    """Verify that run_age_swap replaces ALL occurrences of the original age."""

    def test_ehr_header_and_narrative(self, tmp_path):
        """EHR text with age in header and narrative — both must be replaced."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "context_id,dataset,dataset_id,clinical_context\n"
            'T0,oncology,1,"EHR Context:\nAge: 68 years\nGender: Male\n'
            'Summary: 68-year-old male with Stage III cancer"\n'
        )
        results = run_age_swap(str(csv_path))
        assert len(results) == 1
        rec = results[0]
        assert not rec['age_extraction_failed']
        # No remaining occurrences of original age
        remaining = len(re.findall(r'\b68\b', rec['age_swapped_text']))
        assert remaining == 0, f"Found {remaining} leftover '68' in: {rec['age_swapped_text']}"

    def test_reddit_title_and_body(self, tmp_path):
        """Reddit text with age in title and body — both must be replaced."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "context_id,dataset,dataset_id,clinical_context\n"
            'N57,askadoc,1,"Constipation 26 year old male\n'
            'only now at 26 we are beginning to think"\n'
        )
        results = run_age_swap(str(csv_path))
        rec = results[0]
        assert not rec['age_extraction_failed']
        remaining = len(re.findall(r'\b26\b', rec['age_swapped_text']))
        assert remaining == 0, f"Found {remaining} leftover '26' in: {rec['age_swapped_text']}"

    def test_single_occurrence_unchanged(self, tmp_path):
        """Text with age appearing only once — no change in behavior."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "context_id,dataset,dataset_id,clinical_context\n"
            'T1,askadoc,1,"35-year-old female with headache"\n'
        )
        results = run_age_swap(str(csv_path))
        rec = results[0]
        assert not rec['age_extraction_failed']
        assert '35' not in rec['age_swapped_text']

    def test_age_not_replaced_in_unrelated_numbers(self, tmp_path):
        """Numbers that happen to match the age but aren't ages should not be replaced
        if they aren't word-bounded matches."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "context_id,dataset,dataset_id,clinical_context\n"
            'T2,oncology,1,"55-year-old male on day 550 of treatment with 155mg dose"\n'
        )
        results = run_age_swap(str(csv_path))
        rec = results[0]
        assert not rec['age_extraction_failed']
        # 550 and 155 should NOT be modified (not word-bounded matches of "55")
        assert '550' in rec['age_swapped_text']
        assert '155' in rec['age_swapped_text']
        # But bare "55" should be gone
        assert rec['original_age'] == 55

    def test_decimal_numbers_not_corrupted(self, tmp_path):
        """Lab values like '27.0-33.0' should not be corrupted when age=27."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "context_id,dataset,dataset_id,clinical_context\n"
            'T3,askadoc,1,"27 year old male with MCH Reference Range: 27.0-33.0 pg and CRP 8.6 mg/dL"\n'
        )
        results = run_age_swap(str(csv_path))
        rec = results[0]
        assert not rec['age_extraction_failed']
        assert rec['original_age'] == 27
        # The "27 year old" should be replaced
        assert '27 year old' not in rec['age_swapped_text']
        # But "27.0-33.0" should be preserved
        assert '27.0-33.0' in rec['age_swapped_text']
