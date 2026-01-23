# ABOUTME: Generate calibrated baseline paraphrases for MedPerturb perturbations
# ABOUTME: Creates token-matched paraphrases to test perturbation-specific vs general sensitivity

import sys
sys.path.insert(0, '/scratch/yang.zih/cot_faithfulness/src')

from token_edit_distance import levenshtein_distance


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
