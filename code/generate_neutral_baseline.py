# ABOUTME: Generate neutral sentence baseline dataset for MedPerturb
# ABOUTME: Prepends a fixed clinically-neutral sentence to each original vignette

import argparse
import numpy as np
import pandas as pd

NEUTRAL_SENTENCE = "A family member came with me to the appointment."


def prepend_neutral_sentence(text: str, sentence: str = NEUTRAL_SENTENCE) -> str:
    """Prepend the neutral sentence to the given text, separated by a space."""
    return sentence + ' ' + text


def generate_neutral_baseline_dataset(
    dataset_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Generate a dataset with neutral sentence baselines.

    Reads data.csv, filters out conversational rows, copies non-conversational
    originals and perturbations, and creates neutral baseline rows by prepending
    a fixed sentence to each original.

    Args:
        dataset_path: Path to data.csv
        output_path: Path to write the output CSV

    Returns:
        pd.DataFrame: The generated dataset
    """
    df = pd.read_csv(dataset_path)

    # Filter out conversational rows
    non_conv = df[df['dataset'] != 'conversational'].copy()

    # Identify originals (dataset_id == 1)
    originals = non_conv[non_conv['dataset_id'] == 1]

    # Identify model output columns (all columns with model prefixes)
    model_prefixes = ('GPT4_', 'LLAMA3_', 'LLAMA3-70_', 'PALMYRA-MED_')
    model_cols = [c for c in df.columns if c.startswith(model_prefixes)]

    # Create neutral baseline rows from originals
    baseline_rows = []
    for i, (_, orig_row) in enumerate(originals.iterrows()):
        row = orig_row.copy()
        row['Index'] = 800 + i
        row['dataset_id'] = 6
        row['clinical_context'] = prepend_neutral_sentence(orig_row['clinical_context'])
        for col in model_cols:
            row[col] = np.nan
        baseline_rows.append(row)

    baselines_df = pd.DataFrame(baseline_rows, columns=df.columns)

    # Concatenate: non-conversational originals + perturbations + neutral baselines
    result_df = pd.concat([non_conv, baselines_df], ignore_index=True)

    result_df.to_csv(output_path, index=False)

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate neutral sentence baseline dataset"
    )
    parser.add_argument('--dataset', type=str, default='data.csv',
                        help='Path to original dataset')
    parser.add_argument('--output', type=str, default='data_with_neutral_baseline.csv',
                        help='Output path for dataset with neutral baselines')

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    result_df = generate_neutral_baseline_dataset(args.dataset, args.output)

    n_non_conv = len(result_df[result_df['dataset_id'] != 6])
    n_baseline = len(result_df[result_df['dataset_id'] == 6])
    print(f"\nNeutral baseline dataset created:")
    print(f"  Non-conversational rows: {n_non_conv}")
    print(f"  Neutral baseline rows: {n_baseline}")
    print(f"  Total rows: {len(result_df)}")
    print(f"Saved to: {args.output}")
