# Calibrated Baselines Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 600 calibrated baseline paraphrases to MedPerturb dataset and evaluate with LLAMA3 models to enable perturbation vs baseline comparison.

**Architecture:** Four-step pipeline: (1) Generate calibrated paraphrases matching each perturbation's token change %, (2) Create extended dataset with baseline rows, (3) Evaluate models on baselines, (4) Bootstrap MI analysis comparing perturbation vs baseline effects.

**Tech Stack:** Python 3.11, pandas, asyncio (300 concurrent OpenAI calls), HuggingFace transformers, pytest, SLURM (4× H200 GPUs)

---

## Task 1: Create generate_baselines_v2.py with Correct Data Flow

**Files:**
- Create: `code/generate_baselines_v2.py`
- Test: `tests/unit/test_generate_baselines_v2.py`
- Reference: `docs/plans/2026-01-29-medperturb-baseline-design.md`

**Step 1: Write the failing test for find_original function**

Create `tests/unit/test_generate_baselines_v2.py`:

```python
# ABOUTME: Tests for generate_baselines_v2.py with correct dataset_id/context_id data flow
# ABOUTME: Verifies baseline generation uses perturbation rows to find originals

import pytest
import pandas as pd
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')


class TestFindOriginal:
    """Tests for finding original row given a perturbation row."""

    def test_find_original_returns_matching_row(self):
        """find_original must return row with same dataset+context_id and dataset_id=1."""
        from generate_baselines_v2 import find_original

        df = pd.DataFrame({
            'Index': [0, 1, 2, 3],
            'dataset': ['askadoc', 'askadoc', 'askadoc', 'askadoc'],
            'dataset_id': [1, 2, 1, 2],
            'context_id': ['N75', 'N75', 'N70', 'N70'],
            'clinical_context': ['orig A', 'pert A', 'orig B', 'pert B']
        })

        # Find original for perturbation at Index=1
        pert_row = df[df['Index'] == 1].iloc[0]
        original = find_original(df, pert_row)

        assert original['Index'] == 0
        assert original['dataset_id'] == 1
        assert original['context_id'] == 'N75'
        assert original['clinical_context'] == 'orig A'

    def test_find_original_handles_different_datasets(self):
        """find_original must match on dataset field, not just context_id."""
        from generate_baselines_v2 import find_original

        # Same context_id but different dataset sources
        df = pd.DataFrame({
            'Index': [0, 1, 2, 3],
            'dataset': ['askadoc', 'askadoc', 'oncqa', 'oncqa'],
            'dataset_id': [1, 2, 1, 2],
            'context_id': ['33', '33', '33', '33'],  # Same context_id, different dataset
            'clinical_context': ['askadoc orig', 'askadoc pert', 'oncqa orig', 'oncqa pert']
        })

        # Find original for oncqa perturbation
        pert_row = df[df['Index'] == 3].iloc[0]
        original = find_original(df, pert_row)

        assert original['dataset'] == 'oncqa'
        assert original['clinical_context'] == 'oncqa orig'

    def test_find_original_raises_for_missing(self):
        """find_original must raise KeyError if no matching original exists."""
        from generate_baselines_v2 import find_original

        df = pd.DataFrame({
            'Index': [0, 1],
            'dataset': ['askadoc', 'oncqa'],
            'dataset_id': [1, 2],
            'context_id': ['N75', 'N99'],
            'clinical_context': ['orig', 'pert without orig']
        })

        pert_row = df[df['Index'] == 1].iloc[0]

        with pytest.raises(KeyError):
            find_original(df, pert_row)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_generate_baselines_v2.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'generate_baselines_v2'"

**Step 3: Write minimal implementation**

Create `code/generate_baselines_v2.py`:

```python
# ABOUTME: Generate calibrated baselines for MedPerturb with correct data flow
# ABOUTME: Uses dataset_id/context_id structure to link perturbations to originals

import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/src')

import pandas as pd


def find_original(df: pd.DataFrame, pert_row: pd.Series) -> pd.Series:
    """
    Find the original row corresponding to a perturbation row.

    Args:
        df: Full dataset DataFrame
        pert_row: A perturbation row (dataset_id in 2,3,4,5)

    Returns:
        pd.Series: The matching original row (dataset_id=1)

    Raises:
        KeyError: If no matching original found
    """
    matches = df[
        (df['dataset'] == pert_row['dataset']) &
        (df['context_id'] == pert_row['context_id']) &
        (df['dataset_id'] == 1)
    ]

    if len(matches) == 0:
        raise KeyError(
            f"No original found for dataset={pert_row['dataset']}, "
            f"context_id={pert_row['context_id']}"
        )

    return matches.iloc[0]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_generate_baselines_v2.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add code/generate_baselines_v2.py tests/unit/test_generate_baselines_v2.py
git commit -m "feat: add find_original function for baseline generation"
```

---

## Task 2: Add Baseline Generation Logic

**Files:**
- Modify: `code/generate_baselines_v2.py`
- Modify: `tests/unit/test_generate_baselines_v2.py`

**Step 1: Write the failing test for generate_baselines_for_perturbations**

Add to `tests/unit/test_generate_baselines_v2.py`:

```python
import asyncio
import json


class TestGenerateBaselinesForPerturbations:
    """Tests for the main baseline generation function."""

    def test_generates_baseline_for_each_perturbation(self, tmp_path):
        """Must generate one baseline per perturbation row."""
        from generate_baselines_v2 import generate_baselines_for_perturbations

        df = pd.DataFrame({
            'Index': [0, 1, 2],
            'dataset': ['askadoc', 'askadoc', 'askadoc'],
            'dataset_id': [1, 2, 3],
            'context_id': ['N75', 'N75', 'N75'],
            'clinical_context': ['Original text here', 'Gender swapped text', 'Gender removed text']
        })

        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text.split())))

        # Mock API call that returns paraphrase
        async def mock_api(original, target_pct, tokenizer, openai_client, max_retries, tolerance):
            return {
                'paraphrase': f'Paraphrased: {original[:20]}',
                'actual_pct': target_pct,
                'deviation': 0.1,
                'retries_used': 0
            }

        output_path = tmp_path / 'baselines.json'

        baselines = asyncio.run(generate_baselines_for_perturbations(
            df=df,
            tokenizer=MockTokenizer(),
            output_path=str(output_path),
            max_concurrent=10,
            _paraphrase_fn=mock_api
        ))

        # Should have 2 baselines (for dataset_id 2 and 3)
        assert len(baselines) == 2

        # Check structure
        for b in baselines:
            assert 'source_index' in b  # Which perturbation
            assert 'original_index' in b  # Which original
            assert 'baseline_dataset_id' in b  # 6, 7, 8, or 9
            assert 'paraphrase' in b
            assert 'target_pct' in b
            assert 'actual_pct' in b

    def test_baseline_dataset_id_mapping(self, tmp_path):
        """Baseline dataset_id must be perturbation dataset_id + 4."""
        from generate_baselines_v2 import generate_baselines_for_perturbations

        df = pd.DataFrame({
            'Index': [0, 1, 2, 3, 4],
            'dataset': ['askadoc'] * 5,
            'dataset_id': [1, 2, 3, 4, 5],
            'context_id': ['N75'] * 5,
            'clinical_context': ['orig', 'pert2', 'pert3', 'pert4', 'pert5']
        })

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(10))

        async def mock_api(original, target_pct, tokenizer, openai_client, max_retries, tolerance):
            return {'paraphrase': 'p', 'actual_pct': 5.0, 'deviation': 0.1, 'retries_used': 0}

        output_path = tmp_path / 'baselines.json'

        baselines = asyncio.run(generate_baselines_for_perturbations(
            df=df,
            tokenizer=MockTokenizer(),
            output_path=str(output_path),
            max_concurrent=10,
            _paraphrase_fn=mock_api
        ))

        # Check dataset_id mapping: 2->6, 3->7, 4->8, 5->9
        baseline_by_source = {b['source_index']: b for b in baselines}
        assert baseline_by_source[1]['baseline_dataset_id'] == 6  # gender-swap -> baseline-for-gender-swap
        assert baseline_by_source[2]['baseline_dataset_id'] == 7  # gender-remove -> baseline-for-gender-remove
        assert baseline_by_source[3]['baseline_dataset_id'] == 8  # uncertain -> baseline-for-uncertain
        assert baseline_by_source[4]['baseline_dataset_id'] == 9  # colorful -> baseline-for-colorful

    def test_saves_checkpoint_periodically(self, tmp_path):
        """Must save checkpoint during processing."""
        from generate_baselines_v2 import generate_baselines_for_perturbations

        df = pd.DataFrame({
            'Index': list(range(25)),
            'dataset': ['askadoc'] * 25,
            'dataset_id': [1] + [2] * 24,  # 1 original, 24 perturbations
            'context_id': ['N75'] * 25,
            'clinical_context': ['orig'] + [f'pert{i}' for i in range(24)]
        })

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(10))

        checkpoint_calls = []

        async def mock_api(original, target_pct, tokenizer, openai_client, max_retries, tolerance):
            return {'paraphrase': 'p', 'actual_pct': 5.0, 'deviation': 0.1, 'retries_used': 0}

        output_path = tmp_path / 'baselines.json'

        # Monkey-patch json.dump to track checkpoint saves
        original_dump = json.dump
        def tracking_dump(obj, f, **kwargs):
            checkpoint_calls.append(len(obj))
            return original_dump(obj, f, **kwargs)

        json.dump = tracking_dump
        try:
            asyncio.run(generate_baselines_for_perturbations(
                df=df,
                tokenizer=MockTokenizer(),
                output_path=str(output_path),
                max_concurrent=5,
                checkpoint_freq=10,
                _paraphrase_fn=mock_api
            ))
        finally:
            json.dump = original_dump

        # Should have saved checkpoints
        assert len(checkpoint_calls) >= 2, f"Expected at least 2 checkpoints, got {len(checkpoint_calls)}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_generate_baselines_v2.py::TestGenerateBaselinesForPerturbations -v`
Expected: FAIL with "cannot import name 'generate_baselines_for_perturbations'"

**Step 3: Write minimal implementation**

Add to `code/generate_baselines_v2.py`:

```python
import asyncio
import json
from tqdm.asyncio import tqdm as atqdm

from generate_baselines import compute_token_change_percent
from calibrated_paraphrase import generate_calibrated_paraphrase_async


async def generate_baselines_for_perturbations(
    df: pd.DataFrame,
    tokenizer,
    output_path: str,
    openai_client=None,
    max_concurrent: int = 300,
    checkpoint_freq: int = 10,
    max_retries: int = 10,
    tolerance: float = 0.5,
    _paraphrase_fn=None
) -> list:
    """
    Generate calibrated baselines for all perturbation rows.

    For each perturbation (dataset_id 2-5):
    1. Find corresponding original (same dataset + context_id, dataset_id=1)
    2. Compute token change % between original and perturbation
    3. Generate calibrated paraphrase of original matching that %
    4. Store with metadata linking to source perturbation

    Args:
        df: Dataset with originals and perturbations
        tokenizer: HuggingFace tokenizer for token change calculation
        output_path: Path to save baselines JSON
        openai_client: AsyncOpenAI client (creates one if None)
        max_concurrent: Maximum concurrent API requests
        checkpoint_freq: Save checkpoint every N baselines
        max_retries: Max retries for calibrated paraphrasing
        tolerance: Acceptable deviation from target %
        _paraphrase_fn: Optional mock for testing

    Returns:
        list: Baseline dictionaries
    """
    if openai_client is None and _paraphrase_fn is None:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI()

    # Get perturbation rows only (dataset_id 2-5)
    perturbation_rows = df[df['dataset_id'].isin([2, 3, 4, 5])]

    baselines = []
    baselines_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    completed_lock = asyncio.Lock()

    async def process_perturbation(pert_row):
        nonlocal completed_count

        # Find original
        original = find_original(df, pert_row)

        # Compute token change %
        target_pct = compute_token_change_percent(
            original['clinical_context'],
            pert_row['clinical_context'],
            tokenizer
        )

        # Generate calibrated paraphrase
        async with semaphore:
            if _paraphrase_fn is not None:
                result = await _paraphrase_fn(
                    original['clinical_context'],
                    target_pct,
                    tokenizer,
                    openai_client,
                    max_retries,
                    tolerance
                )
            else:
                result = await generate_calibrated_paraphrase_async(
                    question=original['clinical_context'],
                    target_pct=target_pct,
                    tokenizer=tokenizer,
                    openai_client=openai_client,
                    max_retries=max_retries,
                    tolerance=tolerance
                )

        # Build baseline entry
        baseline = {
            'source_index': int(pert_row['Index']),
            'original_index': int(original['Index']),
            'dataset': pert_row['dataset'],
            'context_id': pert_row['context_id'],
            'perturbation_type': int(pert_row['dataset_id']),
            'baseline_dataset_id': int(pert_row['dataset_id']) + 4,
            'paraphrase': result['paraphrase'],
            'target_pct': target_pct,
            'actual_pct': result['actual_pct'],
            'deviation': result['deviation'],
            'retries_used': result['retries_used']
        }

        async with baselines_lock:
            baselines.append(baseline)

        # Checkpoint
        async with completed_lock:
            completed_count += 1
            if completed_count % checkpoint_freq == 0:
                async with baselines_lock:
                    with open(output_path, 'w') as f:
                        json.dump(baselines, f, indent=2)

    # Create tasks for all perturbations
    tasks = [process_perturbation(row) for _, row in perturbation_rows.iterrows()]

    # Run with progress bar
    await atqdm.gather(*tasks, desc="Generating baselines")

    # Final save
    with open(output_path, 'w') as f:
        json.dump(baselines, f, indent=2)

    return baselines
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_generate_baselines_v2.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add code/generate_baselines_v2.py tests/unit/test_generate_baselines_v2.py
git commit -m "feat: add generate_baselines_for_perturbations with correct data flow"
```

---

## Task 3: Add CLI for generate_baselines_v2.py

**Files:**
- Modify: `code/generate_baselines_v2.py`

**Step 1: No test needed for CLI (integration tested manually)**

**Step 2: Add CLI**

Add to end of `code/generate_baselines_v2.py`:

```python
import argparse
from transformers import AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate calibrated baselines for MedPerturb perturbations"
    )
    parser.add_argument('--dataset', type=str, default='data.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--output', type=str, default='results/baselines_v2.json',
                        help='Output JSON path')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model for tokenizer')
    parser.add_argument('--max_concurrent', type=int, default=300,
                        help='Maximum concurrent API requests')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Save checkpoint every N baselines')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Limit perturbations to process (for testing)')

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"  Total rows: {len(df)}")
    print(f"  Originals (dataset_id=1): {len(df[df['dataset_id'] == 1])}")
    print(f"  Perturbations (dataset_id 2-5): {len(df[df['dataset_id'].isin([2,3,4,5])])}")

    # Optionally limit for testing
    if args.sample_size:
        # Keep all originals, limit perturbations
        originals = df[df['dataset_id'] == 1]
        perturbations = df[df['dataset_id'].isin([2, 3, 4, 5])].head(args.sample_size)
        df = pd.concat([originals, perturbations])
        print(f"  Limited to {len(perturbations)} perturbations (test mode)")

    print(f"\nGenerating baselines with {args.max_concurrent} concurrent requests...")

    baselines = asyncio.run(generate_baselines_for_perturbations(
        df=df,
        tokenizer=tokenizer,
        output_path=args.output,
        max_concurrent=args.max_concurrent,
        checkpoint_freq=args.checkpoint_freq
    ))

    print(f"\nDone! Generated {len(baselines)} baselines")
    print(f"Saved to: {args.output}")
```

**Step 3: Commit**

```bash
git add code/generate_baselines_v2.py
git commit -m "feat: add CLI for generate_baselines_v2.py"
```

---

## Task 4: Create create_extended_dataset.py

**Files:**
- Create: `code/create_extended_dataset.py`
- Test: `tests/unit/test_create_extended_dataset.py`

**Step 1: Write the failing test**

Create `tests/unit/test_create_extended_dataset.py`:

```python
# ABOUTME: Tests for create_extended_dataset.py
# ABOUTME: Verifies correct merging of baselines into dataset with proper schema

import pytest
import pandas as pd
import json
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')


class TestCreateExtendedDataset:
    """Tests for creating extended dataset with baselines."""

    def test_appends_baseline_rows(self, tmp_path):
        """Must append baseline rows to original dataset."""
        from create_extended_dataset import create_extended_dataset

        # Create minimal original dataset
        original_df = pd.DataFrame({
            'Index': [0, 1],
            'dataset': ['askadoc', 'askadoc'],
            'dataset_id': [1, 2],
            'context_id': ['N75', 'N75'],
            'original_patient_gender': ['F', 'F'],
            'clinical_context': ['original text', 'perturbed text'],
            'LLAMA3_MANAGE': [1, 0],
            'LLAMA3_VISIT': [0, 1],
            'LLAMA3_RESOURCE': [1, 1],
        })

        # Create baselines
        baselines = [{
            'source_index': 1,
            'original_index': 0,
            'dataset': 'askadoc',
            'context_id': 'N75',
            'perturbation_type': 2,
            'baseline_dataset_id': 6,
            'paraphrase': 'baseline paraphrase text',
            'target_pct': 15.0,
            'actual_pct': 14.8,
            'deviation': 0.2,
            'retries_used': 1
        }]

        dataset_path = tmp_path / 'data.csv'
        baselines_path = tmp_path / 'baselines.json'
        output_path = tmp_path / 'data_with_baselines.csv'

        original_df.to_csv(dataset_path, index=False)
        with open(baselines_path, 'w') as f:
            json.dump(baselines, f)

        result_df = create_extended_dataset(
            str(dataset_path),
            str(baselines_path),
            str(output_path)
        )

        # Should have 3 rows: 2 original + 1 baseline
        assert len(result_df) == 3

        # Baseline row should have correct values
        baseline_row = result_df[result_df['dataset_id'] == 6].iloc[0]
        assert baseline_row['Index'] == 2  # New index = 800 + position, but for small test just next
        assert baseline_row['clinical_context'] == 'baseline paraphrase text'
        assert baseline_row['dataset'] == 'askadoc'
        assert baseline_row['context_id'] == 'N75'

    def test_baseline_index_starts_at_800(self, tmp_path):
        """Baseline Index values must start at 800."""
        from create_extended_dataset import create_extended_dataset

        # Create dataset with indices 0-799
        original_df = pd.DataFrame({
            'Index': list(range(800)),
            'dataset': ['askadoc'] * 800,
            'dataset_id': [1] * 200 + [2] * 200 + [3] * 200 + [4] * 100 + [5] * 100,
            'context_id': [f'N{i}' for i in range(800)],
            'original_patient_gender': ['F'] * 800,
            'clinical_context': [f'text {i}' for i in range(800)],
        })

        baselines = [{
            'source_index': 200,
            'original_index': 0,
            'dataset': 'askadoc',
            'context_id': 'N0',
            'perturbation_type': 2,
            'baseline_dataset_id': 6,
            'paraphrase': 'baseline text',
            'target_pct': 10.0,
            'actual_pct': 10.0,
            'deviation': 0.0,
            'retries_used': 0
        }]

        dataset_path = tmp_path / 'data.csv'
        baselines_path = tmp_path / 'baselines.json'
        output_path = tmp_path / 'data_with_baselines.csv'

        original_df.to_csv(dataset_path, index=False)
        with open(baselines_path, 'w') as f:
            json.dump(baselines, f)

        result_df = create_extended_dataset(
            str(dataset_path),
            str(baselines_path),
            str(output_path)
        )

        baseline_row = result_df[result_df['dataset_id'] == 6].iloc[0]
        assert baseline_row['Index'] == 800

    def test_model_columns_empty_for_baselines(self, tmp_path):
        """Model output columns must be empty/NaN for baseline rows."""
        from create_extended_dataset import create_extended_dataset

        original_df = pd.DataFrame({
            'Index': [0, 1],
            'dataset': ['askadoc', 'askadoc'],
            'dataset_id': [1, 2],
            'context_id': ['N75', 'N75'],
            'original_patient_gender': ['F', 'F'],
            'clinical_context': ['orig', 'pert'],
            'LLAMA3_MANAGE': [1, 0],
            'LLAMA3-70_MANAGE': [1, 1],
        })

        baselines = [{
            'source_index': 1,
            'original_index': 0,
            'dataset': 'askadoc',
            'context_id': 'N75',
            'perturbation_type': 2,
            'baseline_dataset_id': 6,
            'paraphrase': 'baseline',
            'target_pct': 10.0,
            'actual_pct': 10.0,
            'deviation': 0.0,
            'retries_used': 0
        }]

        dataset_path = tmp_path / 'data.csv'
        baselines_path = tmp_path / 'baselines.json'
        output_path = tmp_path / 'data_with_baselines.csv'

        original_df.to_csv(dataset_path, index=False)
        with open(baselines_path, 'w') as f:
            json.dump(baselines, f)

        result_df = create_extended_dataset(
            str(dataset_path),
            str(baselines_path),
            str(output_path)
        )

        baseline_row = result_df[result_df['dataset_id'] == 6].iloc[0]
        assert pd.isna(baseline_row['LLAMA3_MANAGE'])
        assert pd.isna(baseline_row['LLAMA3-70_MANAGE'])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_create_extended_dataset.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'create_extended_dataset'"

**Step 3: Write minimal implementation**

Create `code/create_extended_dataset.py`:

```python
# ABOUTME: Create extended dataset by appending baseline rows
# ABOUTME: Baseline rows have dataset_id 6-9 and empty model output columns

import pandas as pd
import json
import argparse


def create_extended_dataset(
    dataset_path: str,
    baselines_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Create extended dataset by appending baseline rows.

    Args:
        dataset_path: Path to original data.csv
        baselines_path: Path to baselines.json
        output_path: Path to save extended dataset

    Returns:
        pd.DataFrame: Extended dataset with baseline rows
    """
    # Load original dataset
    df = pd.read_csv(dataset_path)

    # Load baselines
    with open(baselines_path, 'r') as f:
        baselines = json.load(f)

    # Get column list from original
    columns = df.columns.tolist()

    # Find the starting index for baselines
    start_index = 800  # Per design spec

    # Create baseline rows
    baseline_rows = []
    for i, b in enumerate(baselines):
        # Find the original row to copy metadata from
        original_row = df[df['Index'] == b['original_index']].iloc[0]

        row = {
            'Index': start_index + i,
            'dataset': b['dataset'],
            'dataset_id': b['baseline_dataset_id'],
            'context_id': b['context_id'],
            'original_patient_gender': original_row['original_patient_gender'],
            'clinical_context': b['paraphrase'],
        }

        # All other columns are empty (model outputs, clinician ratings, etc.)
        for col in columns:
            if col not in row:
                row[col] = None

        baseline_rows.append(row)

    # Create DataFrame from baseline rows
    baselines_df = pd.DataFrame(baseline_rows, columns=columns)

    # Concatenate
    extended_df = pd.concat([df, baselines_df], ignore_index=True)

    # Save
    extended_df.to_csv(output_path, index=False)

    return extended_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create extended dataset with baseline rows"
    )
    parser.add_argument('--dataset', type=str, default='data.csv',
                        help='Path to original dataset')
    parser.add_argument('--baselines', type=str, default='results/baselines_v2.json',
                        help='Path to baselines JSON')
    parser.add_argument('--output', type=str, default='data_with_baselines.csv',
                        help='Output path for extended dataset')

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    print(f"Loading baselines: {args.baselines}")

    result_df = create_extended_dataset(
        args.dataset,
        args.baselines,
        args.output
    )

    print(f"\nExtended dataset created:")
    print(f"  Original rows: 800")
    print(f"  Baseline rows: {len(result_df) - 800}")
    print(f"  Total rows: {len(result_df)}")
    print(f"Saved to: {args.output}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_create_extended_dataset.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add code/create_extended_dataset.py tests/unit/test_create_extended_dataset.py
git commit -m "feat: add create_extended_dataset.py to merge baselines"
```

---

## Task 5: Create evaluate_baselines.py

**Files:**
- Create: `code/evaluate_baselines.py`
- Test: `tests/unit/test_evaluate_baselines.py`

**Step 1: Write the failing test**

Create `tests/unit/test_evaluate_baselines.py`:

```python
# ABOUTME: Tests for evaluate_baselines.py
# ABOUTME: Verifies baseline-only evaluation with correct model output columns

import pytest
import pandas as pd
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')


class TestEvaluateBaselines:
    """Tests for baseline evaluation."""

    def test_only_evaluates_baseline_rows(self):
        """Must only evaluate rows with dataset_id 6-9."""
        from evaluate_baselines import get_baseline_rows

        df = pd.DataFrame({
            'Index': list(range(10)),
            'dataset_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
            'clinical_context': [f'text {i}' for i in range(10)]
        })

        baseline_df = get_baseline_rows(df)

        assert len(baseline_df) == 4
        assert set(baseline_df['dataset_id'].unique()) == {6, 7, 8, 9}

    def test_shard_data_for_gpu(self):
        """Must correctly shard data across GPUs."""
        from evaluate_baselines import shard_data

        df = pd.DataFrame({
            'Index': list(range(100)),
            'clinical_context': [f'text {i}' for i in range(100)]
        })

        # GPU 0 of 4
        shard0 = shard_data(df, gpu_id=0, total_gpus=4)
        assert len(shard0) == 25
        assert list(shard0['Index']) == list(range(0, 100, 4))

        # GPU 1 of 4
        shard1 = shard_data(df, gpu_id=1, total_gpus=4)
        assert len(shard1) == 25
        assert list(shard1['Index']) == list(range(1, 100, 4))

    def test_model_name_to_column_prefix(self):
        """Must map model names to correct column prefixes."""
        from evaluate_baselines import get_column_prefix

        assert get_column_prefix('meta-llama/Llama-3.1-8B-Instruct') == 'LLAMA3'
        assert get_column_prefix('meta-llama/Llama-3.3-70B-Instruct') == 'LLAMA3-70'
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_evaluate_baselines.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'evaluate_baselines'"

**Step 3: Write minimal implementation**

Create `code/evaluate_baselines.py`:

```python
# ABOUTME: Evaluate models on baseline rows only
# ABOUTME: Supports data parallel sharding across multiple GPUs

import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/code')

import pandas as pd
import argparse
import os
from tqdm import tqdm

from evaluate_models import ModelEvaluator


def get_baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to baseline rows only (dataset_id 6-9)."""
    return df[df['dataset_id'].isin([6, 7, 8, 9])].copy()


def shard_data(df: pd.DataFrame, gpu_id: int, total_gpus: int) -> pd.DataFrame:
    """Shard data for data parallel processing."""
    return df.iloc[gpu_id::total_gpus].copy()


def get_column_prefix(model_name: str) -> str:
    """Map model name to column prefix."""
    if '8B' in model_name:
        return 'LLAMA3'
    elif '70B' in model_name:
        return 'LLAMA3-70'
    else:
        raise ValueError(f"Unknown model: {model_name}")


def majority_vote(responses: list) -> int:
    """Return majority vote from list of 0/1 responses."""
    return 1 if sum(responses) > len(responses) / 2 else 0


def evaluate_baselines(
    df: pd.DataFrame,
    model_name: str,
    gpu_id: int = 0,
    total_gpus: int = 1,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_freq: int = 10
) -> pd.DataFrame:
    """
    Evaluate model on baseline rows.

    Args:
        df: Extended dataset with baseline rows
        model_name: Model to evaluate
        gpu_id: This GPU's ID (for sharding)
        total_gpus: Total GPUs (for sharding)
        checkpoint_dir: Directory for checkpoints
        checkpoint_freq: Save every N rows

    Returns:
        pd.DataFrame: Updated with model outputs for baseline rows
    """
    # Get baseline rows and shard
    baseline_df = get_baseline_rows(df)
    shard = shard_data(baseline_df, gpu_id, total_gpus)

    print(f"GPU {gpu_id}/{total_gpus}: Evaluating {len(shard)} baseline rows")

    # Load model
    evaluator = ModelEvaluator(model_name)
    prefix = get_column_prefix(model_name)

    # Checkpoint path
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/baseline_eval_gpu{gpu_id}.csv"

    # Load checkpoint if exists
    completed_indices = set()
    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        completed_indices = set(checkpoint_df['Index'].tolist())
        print(f"  Resuming from checkpoint: {len(completed_indices)} completed")

    # Evaluate
    results = []
    for idx, row in tqdm(shard.iterrows(), total=len(shard), desc=f"GPU {gpu_id}"):
        if row['Index'] in completed_indices:
            continue

        # Evaluate triage questions
        triage_results = evaluator.evaluate_triage(row['clinical_context'])

        result = {
            'Index': row['Index'],
            f'{prefix}_MANAGE': majority_vote(triage_results['MANAGE']),
            f'{prefix}_VISIT': majority_vote(triage_results['VISIT']),
            f'{prefix}_RESOURCE': majority_vote(triage_results['RESOURCE']),
        }
        results.append(result)

        # Checkpoint
        if len(results) % checkpoint_freq == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(checkpoint_path, index=False)

    # Final save
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(checkpoint_path, index=False)

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model on baseline rows"
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Model to evaluate')
    parser.add_argument('--dataset', type=str, default='data_with_baselines.csv',
                        help='Extended dataset path')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID for this process')
    parser.add_argument('--total_gpus', type=int, default=1,
                        help='Total number of GPUs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/baseline_eval',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Save checkpoint every N rows')

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)

    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu_id}/{args.total_gpus}")

    results_df = evaluate_baselines(
        df=df,
        model_name=args.model,
        gpu_id=args.gpu_id,
        total_gpus=args.total_gpus,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq
    )

    print(f"\nCompleted {len(results_df)} evaluations")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_evaluate_baselines.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add code/evaluate_baselines.py tests/unit/test_evaluate_baselines.py
git commit -m "feat: add evaluate_baselines.py for baseline model evaluation"
```

---

## Task 6: Create SLURM Script for Baseline Evaluation

**Files:**
- Create: `slurm/run_baseline_eval.sbatch`

**Step 1: Write SLURM script**

Create `slurm/run_baseline_eval.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline_eval_%j.out
#SBATCH --error=logs/baseline_eval_%j.err
#SBATCH --signal=INT@300

# ABOUTME: SLURM script for baseline evaluation with 4 H200 GPUs
# ABOUTME: Runs data parallel evaluation across GPUs

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
DATASET=${2:-"data_with_baselines.csv"}

echo "======================================"
echo "Baseline Evaluation"
echo "======================================"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "GPUs: 4x H200"
echo "Started at: $(date)"
echo ""

# Setup
source ~/.bashrc
conda activate cot
cd /scratch/yang.zih/cot_faithfulness/MedPerturb

mkdir -p logs
mkdir -p checkpoints/baseline_eval

# Launch 4 parallel processes
for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python code/evaluate_baselines.py \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --gpu_id ${GPU_ID} \
        --total_gpus 4 \
        --checkpoint_dir checkpoints/baseline_eval \
        --checkpoint_freq 10 &
done

# Wait for all to complete
wait

echo ""
echo "======================================"
echo "All GPUs complete!"
echo "Completed at: $(date)"
echo "======================================"

# Merge results
python << EOF
import pandas as pd
import glob

# Load original dataset
df = pd.read_csv('${DATASET}')

# Load checkpoint files
checkpoint_files = glob.glob('checkpoints/baseline_eval/baseline_eval_gpu*.csv')
print(f"Found {len(checkpoint_files)} checkpoint files")

# Merge results
for f in checkpoint_files:
    results = pd.read_csv(f)
    for _, row in results.iterrows():
        idx = df[df['Index'] == row['Index']].index[0]
        for col in results.columns:
            if col != 'Index':
                df.loc[idx, col] = row[col]

# Save
df.to_csv('${DATASET}', index=False)
print(f"Updated {DATASET} with {len(results)} baseline evaluations")
EOF
```

**Step 2: Commit**

```bash
git add slurm/run_baseline_eval.sbatch
git commit -m "feat: add SLURM script for baseline evaluation"
```

---

## Task 7: Create Pipeline Orchestration Script

**Files:**
- Create: `scripts/auto_run_baselines.sh`

**Step 1: Write orchestration script**

Create `scripts/auto_run_baselines.sh`:

```bash
#!/bin/bash
# ABOUTME: Orchestration script for calibrated baseline pipeline
# ABOUTME: Runs Steps 1-3: generate baselines, create dataset, evaluate model

set -e

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
TEST_MODE=false

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

MEDPERTURB_DIR="/scratch/yang.zih/cot_faithfulness/MedPerturb"
cd "${MEDPERTURB_DIR}"

echo "======================================"
echo "Calibrated Baseline Pipeline"
echo "======================================"
echo "Model: ${MODEL}"
echo "Test mode: ${TEST_MODE}"
echo "Started at: $(date)"
echo ""

source ~/.bashrc
conda activate cot

mkdir -p results
mkdir -p logs
mkdir -p checkpoints

# ==========================================
# STEP 1: Generate Calibrated Baselines
# ==========================================
echo "======================================"
echo "STEP 1: Generate Calibrated Baselines"
echo "======================================"

BASELINE_ARGS="--dataset data.csv --output results/baselines_v2.json --model ${MODEL}"
if [ "$TEST_MODE" = true ]; then
    BASELINE_ARGS="${BASELINE_ARGS} --sample_size 5"
fi

if [ -f "results/baselines_v2.json" ]; then
    echo "Baselines file exists, checking completeness..."
    BASELINE_COUNT=$(python -c "import json; print(len(json.load(open('results/baselines_v2.json'))))")
    EXPECTED=600
    if [ "$TEST_MODE" = true ]; then
        EXPECTED=5
    fi
    if [ "$BASELINE_COUNT" -ge "$EXPECTED" ]; then
        echo "Baselines complete (${BASELINE_COUNT}/${EXPECTED})"
    else
        echo "Baselines incomplete (${BASELINE_COUNT}/${EXPECTED}), resuming..."
        srun --partition=177huntington --cpus-per-task=8 --mem=64G --time=14-00:00:00 \
            bash -c "source ~/.bashrc && conda activate cot && python code/generate_baselines_v2.py ${BASELINE_ARGS}"
    fi
else
    echo "Generating calibrated baselines..."
    srun --partition=177huntington --cpus-per-task=8 --mem=64G --time=14-00:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && python code/generate_baselines_v2.py ${BASELINE_ARGS}"
fi

echo ""
echo "Step 1 complete!"
echo ""

# ==========================================
# STEP 2: Create Extended Dataset
# ==========================================
echo "======================================"
echo "STEP 2: Create Extended Dataset"
echo "======================================"

srun --partition=177huntington --cpus-per-task=4 --mem=32G --time=1:00:00 \
    bash -c "source ~/.bashrc && conda activate cot && python code/create_extended_dataset.py \
        --dataset data.csv \
        --baselines results/baselines_v2.json \
        --output data_with_baselines.csv"

echo ""
echo "Step 2 complete!"
echo ""

# ==========================================
# STEP 3: Evaluate Model on Baselines
# ==========================================
echo "======================================"
echo "STEP 3: Evaluate Model on Baselines"
echo "======================================"

if [ "$TEST_MODE" = true ]; then
    # Test mode: single GPU
    srun --partition=177huntington --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=2:00:00 \
        bash -c "source ~/.bashrc && conda activate cot && python code/evaluate_baselines.py \
            --model ${MODEL} \
            --dataset data_with_baselines.csv \
            --gpu_id 0 \
            --total_gpus 1"
else
    # Production: 4x H200 via sbatch
    sbatch slurm/run_baseline_eval.sbatch "${MODEL}" "data_with_baselines.csv"
    echo "Submitted evaluation job. Monitor with: squeue -u \$USER"
fi

echo ""
echo "======================================"
echo "Pipeline launched!"
echo "======================================"
echo "Completed at: $(date)"
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/auto_run_baselines.sh
git add scripts/auto_run_baselines.sh
git commit -m "feat: add auto_run_baselines.sh pipeline orchestration"
```

---

## Task 8: Create Bootstrap MI Analysis

**Files:**
- Create: `case_studies/baseline_analysis.py`
- Test: `tests/unit/test_baseline_analysis.py`

**Step 1: Write the failing test**

Create `tests/unit/test_baseline_analysis.py`:

```python
# ABOUTME: Tests for bootstrap MI analysis
# ABOUTME: Verifies correct MI calculation and bootstrap hypothesis testing

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/case_studies')


class TestMutualInformation:
    """Tests for MI calculation (matching original notebook)."""

    def test_mi_identical_arrays(self):
        """MI of identical arrays should be maximum (entropy of the array)."""
        from baseline_analysis import calculate_mi

        x = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
        mi = calculate_mi(x, x)

        # MI(X,X) = H(X) = entropy of binary with p=0.5 = 1 bit
        assert abs(mi - 1.0) < 0.01

    def test_mi_independent_arrays(self):
        """MI of independent arrays should be near zero."""
        from baseline_analysis import calculate_mi

        np.random.seed(42)
        x = pd.Series(np.random.randint(0, 2, 1000))
        y = pd.Series(np.random.randint(0, 2, 1000))

        mi = calculate_mi(x, y)
        assert mi < 0.05  # Should be close to 0

    def test_mi_non_negative(self):
        """MI must always be non-negative."""
        from baseline_analysis import calculate_mi

        for _ in range(10):
            x = pd.Series(np.random.randint(0, 2, 100))
            y = pd.Series(np.random.randint(0, 2, 100))
            mi = calculate_mi(x, y)
            assert mi >= 0


class TestBootstrapMITest:
    """Tests for bootstrap hypothesis testing."""

    def test_significant_difference_detected(self):
        """Should detect significant difference when perturbation has larger effect."""
        from baseline_analysis import bootstrap_mi_test

        np.random.seed(42)
        n = 200

        # Original responses
        orig = pd.Series(np.random.randint(0, 2, n))

        # Perturbation causes many flips (high MI with original = low change)
        # Wait, MI measures dependency. If pert = orig, MI is high.
        # If pert is random, MI is low.
        # We want: perturbation causes MORE change than baseline
        # More change = LESS correlation = LOWER MI

        # Perturbation: 50% random (less correlated)
        pert = orig.copy()
        flip_mask = np.random.random(n) < 0.5
        pert[flip_mask] = 1 - pert[flip_mask]

        # Baseline: 20% random (more correlated)
        base = orig.copy()
        flip_mask = np.random.random(n) < 0.2
        base[flip_mask] = 1 - base[flip_mask]

        result = bootstrap_mi_test(orig, pert, base, n_bootstrap=500)

        # MI(orig, pert) should be LOWER than MI(orig, base)
        # because perturbation causes more change
        assert result['mi_perturbation'] < result['mi_baseline']
        assert result['observed_diff'] < 0

    def test_returns_confidence_interval(self):
        """Must return CI bounds."""
        from baseline_analysis import bootstrap_mi_test

        orig = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        pert = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        base = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])

        result = bootstrap_mi_test(orig, pert, base, n_bootstrap=100)

        assert 'ci_low' in result
        assert 'ci_high' in result
        assert result['ci_low'] <= result['ci_high']

    def test_returns_p_value(self):
        """Must return p-value between 0 and 1."""
        from baseline_analysis import bootstrap_mi_test

        orig = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        pert = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        base = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])

        result = bootstrap_mi_test(orig, pert, base, n_bootstrap=100)

        assert 0 <= result['p_value'] <= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_baseline_analysis.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'baseline_analysis'"

**Step 3: Write minimal implementation**

Create `case_studies/baseline_analysis.py`:

```python
# ABOUTME: Bootstrap MI analysis for perturbation vs baseline comparison
# ABOUTME: Tests whether perturbations have specific effects beyond general text changes

import numpy as np
import pandas as pd
import argparse


def calculate_mi(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate mutual information between two arrays.

    Exact implementation from case_study1.ipynb.

    Args:
        x: First array (binary 0/1)
        y: Second array (binary 0/1)

    Returns:
        float: Mutual information in bits
    """
    # Create joint distribution
    joint = pd.crosstab(x, y, normalize=True)
    # Calculate marginal distributions
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    # Calculate mutual information
    mi = 0
    for i in joint.index:
        for j in joint.columns:
            if joint.loc[i, j] > 0:
                mi += joint.loc[i, j] * np.log2(joint.loc[i, j] / (p_x[i] * p_y[j]))
    return mi


def bootstrap_mi_test(
    orig: pd.Series,
    pert: pd.Series,
    base: pd.Series,
    n_bootstrap: int = 1000
) -> dict:
    """
    Bootstrap hypothesis test comparing MI(orig,pert) vs MI(orig,base).

    Args:
        orig: Original responses
        pert: Perturbation responses
        base: Baseline responses
        n_bootstrap: Number of bootstrap iterations

    Returns:
        dict: {
            'mi_perturbation': float,
            'mi_baseline': float,
            'observed_diff': float,
            'ci_low': float,
            'ci_high': float,
            'p_value': float
        }
    """
    n = len(orig)
    diffs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        mi_pert = calculate_mi(
            orig.iloc[indices].reset_index(drop=True),
            pert.iloc[indices].reset_index(drop=True)
        )
        mi_base = calculate_mi(
            orig.iloc[indices].reset_index(drop=True),
            base.iloc[indices].reset_index(drop=True)
        )
        diffs.append(mi_pert - mi_base)

    # Observed difference
    observed_diff = calculate_mi(orig, pert) - calculate_mi(orig, base)

    # 95% CI
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)

    # Two-tailed p-value
    if observed_diff >= 0:
        p_value = np.mean(np.array(diffs) <= 0) * 2
    else:
        p_value = np.mean(np.array(diffs) >= 0) * 2
    p_value = min(p_value, 1.0)

    return {
        'mi_perturbation': calculate_mi(orig, pert),
        'mi_baseline': calculate_mi(orig, base),
        'observed_diff': observed_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_value
    }


def run_analysis(dataset_path: str, output_path: str):
    """
    Run full bootstrap MI analysis.

    Args:
        dataset_path: Path to data_with_baselines.csv
        output_path: Path for output Excel file
    """
    df = pd.read_csv(dataset_path)

    # Mapping
    perturbation_types = {
        2: 'gender_swap',
        3: 'gender_remove',
        4: 'uncertain',
        5: 'colorful'
    }
    baseline_mapping = {2: 6, 3: 7, 4: 8, 5: 9}

    models = ['LLAMA3', 'LLAMA3-70']
    tasks = ['MANAGE', 'VISIT', 'RESOURCE']

    results = []

    for pert_id, pert_name in perturbation_types.items():
        base_id = baseline_mapping[pert_id]

        for model in models:
            for task in tasks:
                col = f'{model}_{task}'

                # Get aligned data by context_id
                originals = df[df['dataset_id'] == 1].set_index(['dataset', 'context_id'])
                perturbations = df[df['dataset_id'] == pert_id].set_index(['dataset', 'context_id'])
                baselines = df[df['dataset_id'] == base_id].set_index(['dataset', 'context_id'])

                # Find common context_ids
                common_idx = originals.index.intersection(perturbations.index).intersection(baselines.index)

                if len(common_idx) == 0:
                    print(f"Warning: No common cases for {pert_name}, {model}, {task}")
                    continue

                orig_vals = originals.loc[common_idx, col].reset_index(drop=True)
                pert_vals = perturbations.loc[common_idx, col].reset_index(drop=True)
                base_vals = baselines.loc[common_idx, col].reset_index(drop=True)

                # Skip if any NaN
                if orig_vals.isna().any() or pert_vals.isna().any() or base_vals.isna().any():
                    print(f"Warning: NaN values for {pert_name}, {model}, {task}")
                    continue

                # Run bootstrap test
                test_result = bootstrap_mi_test(orig_vals, pert_vals, base_vals)

                results.append({
                    'perturbation_type': pert_name,
                    'model': model,
                    'task': task,
                    'n_cases': len(common_idx),
                    **test_result
                })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for _, row in results_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['perturbation_type']:15} {row['model']:10} {row['task']:10} "
              f"diff={row['observed_diff']:+.4f} p={row['p_value']:.4f} {sig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap MI analysis for perturbation vs baseline"
    )
    parser.add_argument('--dataset', type=str, default='data_with_baselines.csv',
                        help='Path to extended dataset')
    parser.add_argument('--output', type=str, default='results/baseline_analysis.xlsx',
                        help='Output Excel path')

    args = parser.parse_args()

    run_analysis(args.dataset, args.output)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_baseline_analysis.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add case_studies/baseline_analysis.py tests/unit/test_baseline_analysis.py
git commit -m "feat: add bootstrap MI analysis for perturbation vs baseline"
```

---

## Task 9: Run Full Test Suite and Final Commit

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short -m "not gpu"
```

Expected: All tests pass

**Step 2: Final commit**

```bash
git add -A
git status
git commit -m "feat: complete calibrated baseline pipeline implementation"
```

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | `generate_baselines_v2.py`, test | find_original function |
| 2 | `generate_baselines_v2.py`, test | Main generation logic |
| 3 | `generate_baselines_v2.py` | CLI |
| 4 | `create_extended_dataset.py`, test | Merge baselines into dataset |
| 5 | `evaluate_baselines.py`, test | Model evaluation on baselines |
| 6 | `run_baseline_eval.sbatch` | SLURM script |
| 7 | `auto_run_baselines.sh` | Pipeline orchestration |
| 8 | `baseline_analysis.py`, test | Bootstrap MI analysis |
| 9 | — | Final test suite and commit |
