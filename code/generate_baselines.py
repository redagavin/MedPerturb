# ABOUTME: Generate calibrated baseline paraphrases for MedPerturb perturbations
# ABOUTME: Creates token-matched paraphrases to test perturbation-specific vs general sensitivity

import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/src')

import pandas as pd
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

from token_edit_distance import levenshtein_distance

PERTURBATION_TYPES = ['gender_swap', 'gender_remove', 'stylistic_uncertain', 'stylistic_colorful']


def compute_token_change_percent(original, perturbed, tokenizer=None, use_words=False):
    """
    Compute token change percentage between original and perturbed text.

    Args:
        original: Original text
        perturbed: Perturbed text
        tokenizer: HuggingFace tokenizer (optional if use_words=True)
        use_words: If True, use word-level comparison (for testing)

    Returns:
        float: Percentage of tokens changed
    """
    if use_words:
        orig_tokens = original.split()
        pert_tokens = perturbed.split()
    else:
        orig_tokens = tokenizer.encode(original, add_special_tokens=False)
        pert_tokens = tokenizer.encode(perturbed, add_special_tokens=False)

    if len(orig_tokens) == 0:
        return 0.0 if len(pert_tokens) == 0 else 100.0

    edit_dist = levenshtein_distance(orig_tokens, pert_tokens)
    return (edit_dist / len(orig_tokens)) * 100


from calibrated_paraphrase import generate_calibrated_paraphrase


def generate_baseline(original_text, target_pct, tokenizer, openai_client=None,
                      max_retries=10, tolerance=0.5):
    """
    Generate a calibrated baseline paraphrase matching target token change.

    Args:
        original_text: Original clinical context
        target_pct: Target token change percentage to match
        tokenizer: HuggingFace tokenizer for the evaluation model
        openai_client: OpenAI client (creates one if None)
        max_retries: Maximum calibration attempts
        tolerance: Acceptable deviation from target (±%)

    Returns:
        dict: {paraphrase, target_pct, actual_pct, deviation, retries_used}
    """
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI()

    result = generate_calibrated_paraphrase(
        question=original_text,
        target_pct=target_pct,
        tokenizer=tokenizer,
        openai_client=openai_client,
        max_retries=max_retries,
        tolerance=tolerance
    )

    return {
        'paraphrase': result['paraphrase'],
        'target_pct': target_pct,
        'actual_pct': result['actual_pct'],
        'deviation': result['deviation'],
        'retries_used': result['retries_used']
    }


def load_dataset_with_perturbations(dataset_path, perturbations_dir):
    """
    Load original dataset and pre-generated perturbations.

    Args:
        dataset_path: Path to original dataset.csv
        perturbations_dir: Directory containing perturbation JSON files

    Returns:
        pd.DataFrame: Dataset with perturbation columns added
    """
    df = pd.read_csv(dataset_path)

    # Load perturbations for each type
    for ptype in PERTURBATION_TYPES:
        pfile = f"{perturbations_dir}/{ptype}.json"
        try:
            with open(pfile, 'r') as f:
                perturbations = json.load(f)
            df[f'{ptype}_text'] = df['context_id'].map(
                lambda cid: perturbations.get(str(cid), {}).get('perturbed', '')
            )
        except FileNotFoundError:
            print(f"Warning: {pfile} not found, skipping {ptype}")
            df[f'{ptype}_text'] = ''

    return df


def generate_all_baselines(df, tokenizer, output_path, openai_client=None):
    """
    Generate calibrated baselines for all samples and perturbations.

    Args:
        df: DataFrame with original and perturbed texts
        tokenizer: HuggingFace tokenizer
        output_path: Path to save results JSON
        openai_client: OpenAI client

    Returns:
        dict: Baselines indexed by context_id and perturbation type
    """
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI()

    baselines = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating baselines"):
        context_id = str(row['context_id'])
        original = row['clinical_context']
        baselines[context_id] = {}

        for ptype in PERTURBATION_TYPES:
            perturbed = row.get(f'{ptype}_text', '')
            if not perturbed:
                continue

            # Compute target token change %
            target_pct = compute_token_change_percent(original, perturbed, tokenizer)

            if target_pct < 0.5:
                # Skip if perturbation is too small
                baselines[context_id][ptype] = {
                    'paraphrase': original,
                    'target_pct': target_pct,
                    'actual_pct': 0.0,
                    'deviation': target_pct,
                    'retries_used': 0,
                    'skipped': True
                }
                continue

            # Generate calibrated baseline
            result = generate_baseline(
                original_text=original,
                target_pct=target_pct,
                tokenizer=tokenizer,
                openai_client=openai_client
            )
            result['skipped'] = False
            baselines[context_id][ptype] = result

        # Save checkpoint every 10 samples
        if (idx + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(baselines, f, indent=2)

    # Final save
    with open(output_path, 'w') as f:
        json.dump(baselines, f, indent=2)

    return baselines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate calibrated baselines for MedPerturb")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset.csv')
    parser.add_argument('--perturbations_dir', type=str, required=True, help='Directory with perturbation JSONs')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model for tokenizer')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Limit number of samples (for testing)')

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading dataset: {args.dataset}")
    df = load_dataset_with_perturbations(args.dataset, args.perturbations_dir)

    # Apply sample size limit if specified
    if args.sample_size:
        df = df.head(args.sample_size)
        print(f"Limited to {len(df)} samples (test mode)")

    print(f"Generating baselines for {len(df)} samples...")
    baselines = generate_all_baselines(df, tokenizer, args.output)

    print(f"Done! Baselines saved to: {args.output}")
