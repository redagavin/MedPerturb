# ABOUTME: Evaluate models on baseline rows only
# ABOUTME: Supports data parallel sharding across multiple GPUs

import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/MedPerturb/.worktrees/calibrated-baselines/code')

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
