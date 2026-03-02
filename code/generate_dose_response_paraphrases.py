# ABOUTME: Generate calibrated paraphrases at multiple token change levels
# ABOUTME: Produces dose-response data showing how token change % affects answer stability

import asyncio
import json
import os
import shutil
import tempfile

import pandas as pd


def load_samples(df):
    """Load non-conversational original samples.

    Args:
        df: DataFrame with data.csv structure

    Returns:
        list of dicts with context_id, dataset, clinical_context
    """
    originals = df[(df['dataset_id'] == 1) & (df['dataset'] != 'conversational')]
    samples = []
    for _, row in originals.iterrows():
        samples.append({
            'context_id': row['context_id'],
            'dataset': row['dataset'],
            'clinical_context': row['clinical_context'],
        })
    return samples


def atomic_save(data, path):
    """Write JSON atomically to prevent corruption on crash."""
    dir_path = os.path.dirname(path) or '.'
    os.makedirs(dir_path, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_path, suffix='.tmp') as f:
        json.dump(data, f, indent=2)
        temp_path = f.name
    shutil.move(temp_path, path)


async def generate_paraphrases(
    samples,
    targets,
    tokenizer,
    openai_client,
    output_path=None,
    max_concurrent=300,
    checkpoint_freq=10,
    max_retries=50,
    tolerance=0.5,
    _paraphrase_fn=None,
):
    """Generate calibrated paraphrases at multiple token change levels.

    Args:
        samples: List of dicts with context_id, dataset, clinical_context
        targets: List of target token change percentages
        tokenizer: HuggingFace tokenizer
        openai_client: AsyncOpenAI client
        output_path: Path for JSON output (enables checkpointing/resume)
        max_concurrent: Maximum concurrent API requests
        checkpoint_freq: Save checkpoint every N paraphrases
        max_retries: Max retries per paraphrase
        tolerance: Acceptable deviation from target %
        _paraphrase_fn: Injectable paraphrase function for testing

    Returns:
        list: Paraphrase result dicts
    """
    if _paraphrase_fn is None:
        from calibrated_paraphrase import generate_calibrated_paraphrase_async
        _paraphrase_fn = generate_calibrated_paraphrase_async

    # Resume from checkpoint
    results = []
    completed_keys = set()
    if output_path and os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
        completed_keys = {(r['context_id'], r['target_pct']) for r in results}
        print(f"  Resuming: {len(results)} paraphrases already generated")

    results_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    completed_lock = asyncio.Lock()

    async def process_one(sample, target_pct):
        nonlocal completed_count

        key = (sample['context_id'], target_pct)
        if key in completed_keys:
            return

        async with semaphore:
            result = await _paraphrase_fn(
                sample['clinical_context'],
                target_pct,
                tokenizer,
                openai_client,
                max_retries,
                tolerance,
            )

        entry = {
            'context_id': sample['context_id'],
            'dataset': sample['dataset'],
            'target_pct': target_pct,
            'actual_pct': result['actual_pct'],
            'deviation': result['deviation'],
            'retries_used': result['retries_used'],
            'paraphrase': result['paraphrase'],
        }

        async with results_lock:
            results.append(entry)

        async with completed_lock:
            completed_count += 1
            if output_path and completed_count % checkpoint_freq == 0:
                async with results_lock:
                    atomic_save(results, output_path)

    tasks = []
    for sample in samples:
        for target in targets:
            tasks.append(process_one(sample, target))

    await asyncio.gather(*tasks)

    if output_path:
        atomic_save(results, output_path)

    return results
