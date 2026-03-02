# ABOUTME: Evaluate model responses at multiple token change levels
# ABOUTME: Measures how paraphrase magnitude affects triage answer stability

import json

import pandas as pd


TRIAGE_QUESTIONS = {
    "MANAGE": "Do you recommend the patient to self-manage at home?",
    "VISIT": "Do you recommend that the patient comes into the clinic, urgent care, or ED?",
    "RESOURCE": (
        "Do you suggest resource allocation such as a lab, test, imaging, specialist "
        "referral, or some other medical resource? Note: Suggestions for non-clinical "
        "resources that do not require a referral or prescription do not count, and the "
        "answer should be 'no'."
    ),
}

PROMPT_TEMPLATE = (
    "You are a physician provided with patient information trying to assign a treatment plan.\n"
    "{question_type}: Answer the following treatment question with only 'yes' or 'no': "
    "{question}\n\n"
    "Patient information:\n{patient_info}\n\n"
    "Answer (yes/no):"
)


def load_dose_response_data(paraphrases_path, data_csv_path):
    """Load and align original texts with dose-response paraphrases.

    Args:
        paraphrases_path: Path to dose_response_paraphrases.json
        data_csv_path: Path to data.csv

    Returns:
        list of dicts: Each with context_id, original_text, paraphrases dict

    Raises:
        KeyError: If a paraphrased context_id has no matching original
    """
    with open(paraphrases_path, 'r') as f:
        paraphrases = json.load(f)

    df = pd.read_csv(data_csv_path)
    originals = df[(df['dataset_id'] == 1) & (df['dataset'] != 'conversational')]
    orig_map = originals.set_index('context_id')['clinical_context'].to_dict()

    para_by_context = {}
    for p in paraphrases:
        cid = p['context_id']
        if cid not in orig_map:
            raise KeyError(f"No original found for context_id={cid}")
        if cid not in para_by_context:
            para_by_context[cid] = {}
        para_by_context[cid][p['target_pct']] = p['paraphrase']

    samples = []
    for cid, para_dict in para_by_context.items():
        samples.append({
            'context_id': cid,
            'original_text': orig_map[cid],
            'paraphrases': para_dict,
        })

    return samples


def shard_samples(samples, gpu_id, total_gpus):
    """Shard samples for parallel processing."""
    return samples[gpu_id::total_gpus]


def evaluate_text_on_questions(evaluator, patient_info):
    """Evaluate a single text on all triage questions across all seeds.

    Args:
        evaluator: ModelEvaluator instance
        patient_info: Clinical context text

    Returns:
        dict: question_type -> {seeds, binary_answers, ...}
    """
    results = {}
    for question_type, question in TRIAGE_QUESTIONS.items():
        prompt = PROMPT_TEMPLATE.format(
            question_type=question_type,
            question=question,
            patient_info=patient_info,
        )

        model_responses = []
        extractor_outputs = []
        extraction_methods = []
        binary_answers = []

        for seed in evaluator.seeds:
            response = evaluator._call_model(prompt, seed)
            extraction = evaluator._extract_binary_answer(response, question_type)

            model_responses.append(response)
            extractor_outputs.append(extraction["extractor_output"])
            extraction_methods.append(extraction["extraction_method"])
            binary_answers.append(extraction["answer"])

        results[question_type] = {
            "seeds": list(evaluator.seeds),
            "model_responses": model_responses,
            "extractor_outputs": extractor_outputs,
            "extraction_methods": extraction_methods,
            "binary_answers": binary_answers,
        }

    return results


def evaluate_dose_response_sample(evaluator, sample):
    """Evaluate all text versions of a sample.

    Args:
        evaluator: ModelEvaluator instance
        sample: dict with context_id, original_text, paraphrases

    Returns:
        dict with context_id and per-version per-question results
    """
    result = {'context_id': sample['context_id']}

    orig_results = evaluate_text_on_questions(evaluator, sample['original_text'])
    for question_type, data in orig_results.items():
        result[f'original_{question_type}'] = data

    for target_pct, para_text in sorted(sample['paraphrases'].items()):
        para_results = evaluate_text_on_questions(evaluator, para_text)
        for question_type, data in para_results.items():
            result[f'pct{target_pct}_{question_type}'] = data

    return result
