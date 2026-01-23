import pytest
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

@pytest.fixture
def sample_clinical_context():
    return "A 45-year-old woman presents with chest pain radiating to her left arm."

@pytest.fixture
def sample_perturbed_context():
    return "A 45-year-old man presents with chest pain radiating to his left arm."
