# ABOUTME: Batch evaluation script for MedPerturb with baseline comparison
# ABOUTME: Evaluates all 9 text versions (original + 4 perturbations + 4 baselines) per sample

import os
import sys
import json
import argparse
import pickle
import time
import pandas as pd
from tqdm import tqdm

# Add directories for imports
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # MedPerturb/code

from evaluate_models import ModelEvaluator

PERTURBATION_TYPES = ['gender_swap', 'gender_remove', 'stylistic_uncertain', 'stylistic_colorful']


def load_all_data(dataset_path, perturbations_dir, baselines_path):
    """
    Load dataset, perturbations, and baselines.

    Returns:
        pd.DataFrame with columns: context_id, clinical_context,
            {ptype}_perturbed, {ptype}_baseline for each perturbation type
    """
    # Load original dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples from dataset")

    # Load perturbations (keyed by Index, the unique row identifier)
    for ptype in PERTURBATION_TYPES:
        pfile = f"{perturbations_dir}/{ptype}.json"
        if os.path.exists(pfile):
            with open(pfile, 'r') as f:
                perturbations = json.load(f)
            df[f'{ptype}_perturbed'] = df['Index'].apply(
                lambda idx: perturbations.get(str(idx), {}).get('perturbed', '')
            )
            print(f"  Loaded {ptype} perturbations")
        else:
            print(f"  Warning: {pfile} not found")
            df[f'{ptype}_perturbed'] = ''

    # Load baselines (keyed by Index, the unique row identifier)
    if os.path.exists(baselines_path):
        with open(baselines_path, 'r') as f:
            baselines = json.load(f)

        for ptype in PERTURBATION_TYPES:
            df[f'{ptype}_baseline'] = df['Index'].apply(
                lambda idx: baselines.get(str(idx), {}).get(ptype, {}).get('paraphrase', '')
            )
        print(f"  Loaded baselines")
    else:
        print(f"  Warning: {baselines_path} not found")
        for ptype in PERTURBATION_TYPES:
            df[f'{ptype}_baseline'] = ''

    return df


def shard_data(df, gpu_id, total_gpus):
    """Shard data for parallel processing."""
    total_samples = len(df)
    samples_per_gpu = total_samples // total_gpus
    start_idx = gpu_id * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if gpu_id < total_gpus - 1 else total_samples

    print(f"GPU {gpu_id}: Processing samples {start_idx} to {end_idx-1} ({end_idx - start_idx} samples)")
    return df.iloc[start_idx:end_idx].copy()


def save_checkpoint(checkpoint_path, results, completed_indices):
    """Save checkpoint to disk."""
    checkpoint_data = {
        'results': results,
        'completed_indices': list(completed_indices)
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"  Checkpoint saved: {len(results)} results")


def load_checkpoint(checkpoint_path):
    """Load checkpoint from disk."""
    if not os.path.exists(checkpoint_path):
        return [], set()

    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)

    results = checkpoint_data.get('results', [])
    completed_indices = set(checkpoint_data.get('completed_indices', []))
    print(f"  Loaded checkpoint: {len(results)} results")
    return results, completed_indices


def mark_gpu_completion(checkpoint_dir, model_short, gpu_id, total_gpus):
    """Mark this GPU as complete."""
    marker_path = f"{checkpoint_dir}/{model_short}_gpu{gpu_id}_of_{total_gpus}_COMPLETE"
    with open(marker_path, 'w') as f:
        f.write(str(time.time()))
    print(f"GPU {gpu_id} marked as complete")


def evaluate_sample(evaluator, sample_id, context_id, texts_dict):
    """
    Evaluate all text versions for a single sample.

    Args:
        evaluator: ModelEvaluator instance
        sample_id: Unique row identifier (Index column from CSV)
        context_id: Patient/case identifier (not unique per row)
        texts_dict: Dict with keys 'original', '{ptype}_perturbed', '{ptype}_baseline'

    Returns:
        dict with evaluation results for all text versions
    """
    result = {'sample_id': sample_id, 'context_id': context_id}

    # Evaluate original
    if texts_dict.get('original'):
        try:
            original_results = evaluator.evaluate_triage(texts_dict['original'])
            for q, responses in original_results.items():
                result[f'original_{q}'] = responses
        except Exception as e:
            print(f"    Error evaluating original: {e}")

    # Evaluate each perturbation and its baseline
    for ptype in PERTURBATION_TYPES:
        # Perturbation
        perturbed_text = texts_dict.get(f'{ptype}_perturbed', '')
        if perturbed_text:
            try:
                pert_results = evaluator.evaluate_triage(perturbed_text)
                for q, responses in pert_results.items():
                    result[f'{ptype}_{q}'] = responses
            except Exception as e:
                print(f"    Error evaluating {ptype} perturbation: {e}")

        # Baseline
        baseline_text = texts_dict.get(f'{ptype}_baseline', '')
        if baseline_text:
            try:
                base_results = evaluator.evaluate_triage(baseline_text)
                for q, responses in base_results.items():
                    result[f'{ptype}_baseline_{q}'] = responses
            except Exception as e:
                print(f"    Error evaluating {ptype} baseline: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for MedPerturb")
    parser.add_argument('--model', type=str, required=True, help='Model to evaluate')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset.csv')
    parser.add_argument('--perturbations_dir', type=str, required=True, help='Directory with perturbation JSONs')
    parser.add_argument('--baselines', type=str, required=True, help='Path to baselines.json')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint every N samples')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID for data sharding')
    parser.add_argument('--total_gpus', type=int, default=1, help='Total GPUs in parallel job')
    parser.add_argument('--sample_size', type=int, default=None, help='Limit samples for testing')

    args = parser.parse_args()

    # Auto-detect SLURM array job parameters
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.gpu_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        args.total_gpus = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        print(f"Detected SLURM array job: GPU {args.gpu_id} of {args.total_gpus}")
    elif args.gpu_id is None:
        args.gpu_id = 0

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Model short name for checkpoint files
    model_short = args.model.split('/')[-1].lower().replace('-', '_')
    checkpoint_path = f"{args.checkpoint_dir}/{model_short}_gpu{args.gpu_id}.pkl"

    print("========================================")
    print("MedPerturb Batch Evaluation")
    print("========================================")
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu_id} of {args.total_gpus}")
    print(f"Checkpoint: {checkpoint_path}")
    print("")

    # Load all data
    print("Loading data...")
    df = load_all_data(args.dataset, args.perturbations_dir, args.baselines)

    # Shard data for parallel processing
    df = shard_data(df, args.gpu_id, args.total_gpus)

    # Apply sample size limit if specified
    if args.sample_size:
        df = df.head(args.sample_size)
        print(f"Limited to {len(df)} samples for testing")

    # Load checkpoint
    results, completed_indices = load_checkpoint(checkpoint_path)

    # Initialize evaluator
    print(f"\nInitializing model: {args.model}")
    evaluator = ModelEvaluator(args.model)

    # Process samples
    print(f"\nProcessing {len(df)} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU {args.gpu_id}"):
        if idx in completed_indices:
            continue

        sample_id = str(row['Index'])
        context_id = str(row['context_id'])

        # Build texts dict
        texts_dict = {
            'original': row['clinical_context']
        }
        for ptype in PERTURBATION_TYPES:
            texts_dict[f'{ptype}_perturbed'] = row.get(f'{ptype}_perturbed', '')
            texts_dict[f'{ptype}_baseline'] = row.get(f'{ptype}_baseline', '')

        # Evaluate
        result = evaluate_sample(evaluator, sample_id, context_id, texts_dict)
        results.append(result)
        completed_indices.add(idx)

        # Checkpoint
        if len(results) % args.checkpoint_freq == 0:
            save_checkpoint(checkpoint_path, results, completed_indices)

    # Final save
    save_checkpoint(checkpoint_path, results, completed_indices)

    # Save JSON output
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Mark completion
    mark_gpu_completion(args.checkpoint_dir, model_short, args.gpu_id, args.total_gpus)

    print("\n========================================")
    print(f"GPU {args.gpu_id} complete: {len(results)} samples evaluated")
    print("========================================")


if __name__ == "__main__":
    main()
