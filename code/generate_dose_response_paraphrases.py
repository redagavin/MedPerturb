# ABOUTME: Generate calibrated paraphrases at multiple token change levels
# ABOUTME: Produces dose-response data showing how token change % affects answer stability

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
