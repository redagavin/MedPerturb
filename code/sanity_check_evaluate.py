# ABOUTME: Evaluation script for gender question sanity check
# ABOUTME: Validates MedPerturb MI pipeline by asking "Is this patient male?" on gender-swapped texts

import pandas as pd


def load_sanity_check_data(dataset_path):
    """Load and align original, gender-swap, and baseline texts.

    Reads data_with_baselines.csv, filters to non-conversational samples,
    and aligns the three text versions by context_id.

    Args:
        dataset_path: Path to data_with_baselines.csv

    Returns:
        list of dicts: Each with context_id, original_text, swap_text, baseline_text
    """
    df = pd.read_csv(dataset_path)

    # Exclude conversational subset (format perturbation, not gender swap)
    df = df[df['dataset'] != 'conversational']

    originals = df[df['dataset_id'] == 1].set_index('context_id')
    swaps = df[df['dataset_id'] == 2].set_index('context_id')
    baselines = df[df['dataset_id'] == 6].set_index('context_id')

    # Only include samples that have all three versions
    common = originals.index.intersection(swaps.index).intersection(baselines.index)

    samples = []
    for cid in common:
        samples.append({
            'context_id': cid,
            'original_text': originals.loc[cid, 'clinical_context'],
            'swap_text': swaps.loc[cid, 'clinical_context'],
            'baseline_text': baselines.loc[cid, 'clinical_context'],
        })

    return samples
