# ABOUTME: Evaluation script for gender question sanity check
# ABOUTME: Validates MedPerturb MI pipeline by asking "Is this patient male?" on gender-swapped texts

import os
import json
import pickle
import shutil
import tempfile
import time
import argparse

import pandas as pd
from tqdm import tqdm

GENDER_QUESTION = "Is this patient male?"

GENDER_PROMPT_TEMPLATE = (
    "You are a physician provided with patient information.\n"
    "GENDER: Answer the following question with only 'yes' or 'no': {question}\n\n"
    "Patient information:\n{patient_info}\n\n"
    "Answer (yes/no):"
)


def build_gender_prompt(patient_info):
    """Build the gender question prompt for a patient.

    Args:
        patient_info: Clinical context text

    Returns:
        str: Complete prompt for model evaluation
    """
    return GENDER_PROMPT_TEMPLATE.format(
        question=GENDER_QUESTION,
        patient_info=patient_info
    )


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


def evaluate_gender_question(evaluator, patient_info):
    """Evaluate the gender question for a single text across all seeds.

    Args:
        evaluator: ModelEvaluator instance (uses _call_model and _extract_binary_answer)
        patient_info: Clinical context text

    Returns:
        list[int]: Binary responses (0/1) for each seed
    """
    prompt = build_gender_prompt(patient_info)
    responses = []
    for seed in evaluator.seeds:
        response = evaluator._call_model(prompt, seed)
        binary = evaluator._extract_binary_answer(response, "GENDER")
        responses.append(binary)
    return responses


def evaluate_sanity_check_sample(evaluator, sample):
    """Evaluate gender question on all three text versions of a sample.

    Args:
        evaluator: ModelEvaluator instance
        sample: dict with context_id, original_text, swap_text, baseline_text

    Returns:
        dict with context_id, original_GENDER, gender_swap_GENDER, gender_swap_baseline_GENDER
    """
    return {
        'context_id': sample['context_id'],
        'original_GENDER': evaluate_gender_question(evaluator, sample['original_text']),
        'gender_swap_GENDER': evaluate_gender_question(evaluator, sample['swap_text']),
        'gender_swap_baseline_GENDER': evaluate_gender_question(evaluator, sample['baseline_text']),
    }


def save_checkpoint(checkpoint_path, results, completed_context_ids):
    """Save checkpoint to disk atomically to prevent corruption on crash."""
    checkpoint_data = {
        'results': results,
        'completed_context_ids': list(completed_context_ids)
    }
    dir_path = os.path.dirname(checkpoint_path) or '.'
    with tempfile.NamedTemporaryFile('wb', delete=False, dir=dir_path, suffix='.tmp') as f:
        pickle.dump(checkpoint_data, f)
        temp_path = f.name
    shutil.move(temp_path, checkpoint_path)


def load_checkpoint(checkpoint_path):
    """Load checkpoint from disk."""
    if not os.path.exists(checkpoint_path):
        return [], set()
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    results = checkpoint_data.get('results', [])
    completed = set(checkpoint_data.get('completed_context_ids', []))
    return results, completed


def shard_samples(samples, gpu_id, total_gpus):
    """Shard samples for parallel processing."""
    return samples[gpu_id::total_gpus]


def main():
    parser = argparse.ArgumentParser(description="Sanity check: gender question evaluation")
    parser.add_argument('--model', type=str, required=True, help='Model to evaluate')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to data_with_baselines.csv')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/sanity_check',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Checkpoint every N samples')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID for sharding')
    parser.add_argument('--total_gpus', type=int, default=1, help='Total GPUs')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Limit samples for testing')

    args = parser.parse_args()

    # Auto-detect SLURM array job
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        args.gpu_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        args.total_gpus = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        print(f"Detected SLURM array job: GPU {args.gpu_id} of {args.total_gpus}")
    elif args.gpu_id is None:
        args.gpu_id = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_short = args.model.split('/')[-1].lower().replace('-', '_')
    checkpoint_path = f"{args.checkpoint_dir}/{model_short}_gpu{args.gpu_id}.pkl"

    print("=" * 40)
    print("Sanity Check: Gender Question Evaluation")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu_id} of {args.total_gpus}")
    print()

    # Load data
    print("Loading data...")
    samples = load_sanity_check_data(args.dataset)
    print(f"  {len(samples)} non-conversational samples")

    # Shard
    samples = shard_samples(samples, args.gpu_id, args.total_gpus)
    print(f"  GPU {args.gpu_id} shard: {len(samples)} samples")

    if args.sample_size:
        samples = samples[:args.sample_size]
        print(f"  Limited to {len(samples)} samples (test mode)")

    # Load checkpoint
    results, completed = load_checkpoint(checkpoint_path)
    if completed:
        print(f"  Resuming: {len(completed)} already completed")

    # Initialize model
    print(f"\nInitializing model: {args.model}")
    from evaluate_models import ModelEvaluator
    evaluator = ModelEvaluator(args.model)

    # Evaluate
    print(f"\nEvaluating {len(samples)} samples...")
    for sample in tqdm(samples, desc=f"GPU {args.gpu_id}"):
        if sample['context_id'] in completed:
            continue

        result = evaluate_sanity_check_sample(evaluator, sample)
        results.append(result)
        completed.add(sample['context_id'])

        if len(results) % args.checkpoint_freq == 0:
            save_checkpoint(checkpoint_path, results, completed)

    # Final save
    save_checkpoint(checkpoint_path, results, completed)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Mark completion
    marker = f"{args.checkpoint_dir}/{model_short}_gpu{args.gpu_id}_of_{args.total_gpus}_COMPLETE"
    with open(marker, 'w') as f:
        f.write(str(time.time()))

    print(f"\nGPU {args.gpu_id} complete: {len(results)} samples evaluated")


if __name__ == "__main__":
    main()
